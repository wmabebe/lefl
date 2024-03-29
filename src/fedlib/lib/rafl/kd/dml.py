"""
Reference:
Source library: A PyTorch model compression library containing easy-to-use methods for knowledge distillation, pruning, and quantization
Authors: Het Shah and Avishree Khare and Neelay Shah and Khizir Siddiqui
ArXiv: https://arxiv.org/abs/2011.14691 
GitHub: https://github.com/SforAiDl/KD_Lib

Improvement: add lr adjustment and lr schedular option
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
import matplotlib.pyplot as plt

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def adjust_learning_rate(optimizer, epoch, initial_lr, n_epoch, lr_type='cos'):
    if lr_type == 'cos':  # cos without warm-up
        lr = 0.5 * initial_lr * (1 + math.cos(math.pi * epoch / n_epoch))
    elif lr_type == 'exp':
        step = 1
        decay = 0.96
        lr = initial_lr * (decay ** (epoch // step))
    elif lr_type == 'fixed':
        lr = initial_lr
    else:
        raise NotImplementedError
    # print('=> lr: {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class DML:
    """
    Implementation of "Deep Mutual Learning" https://arxiv.org/abs/1706.00384

    :param student_cohort (list/tuple): Collection of student models
    :param train_loader (torch.utils.data.DataLoader): Dataloader for training
    :param val_loader (torch.utils.data.DataLoader): Dataloader for validation/testing
    :param student_optimizers (list/tuple): Collection of Pytorch optimizers for training students
    :param loss_fn (torch.nn.Module): Loss Function used for distillation
    :param device (str): Device used for training; 'cpu' for cpu and 'cuda' for gpu
    :param log (bool): True if logging required
    :param logdir (str): Directory for storing logs
    """

    def __init__(
            self,
            student_cohort,
            train_loader,
            val_loader,
            student_optimizers,
            lr_schedulers,
            loss_fn=nn.MSELoss(),
            device="cpu",
            log=False,
            logdir="./Experiments",
    ):
        self.student_cohort = student_cohort
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.student_optimizers = student_optimizers
        self.lr_schedulers = lr_schedulers
        self.loss_fn = loss_fn
        self.log = log
        self.logdir = logdir

        if self.log:
            self.writer = SummaryWriter(logdir)

        try:
            torch.Tensor(0).to(device)
            self.device = device
        except:
            print(
                "Either an invalid device or CUDA is not available. Defaulting to CPU."
            )
            self.device = "cpu"

    def train_students(
            self,
            epochs=20,
            plot_losses=True,
    ):

        for student in self.student_cohort:
            student.train()

        loss_arr = []

        length_of_dataset = len(self.train_loader.dataset)

        num_students = len(self.student_cohort)

        print("\nTraining students...")

        for ep in range(epochs):
            epoch_loss = 0.0
            correct = 0
            # for optimizer in self.student_optimizers:
            #     lr = adjust_learning_rate(optimizer, ep, self.lr, epochs, lr_type='cos')

            for (data, label) in self.train_loader:

                data = data.to(self.device)
                label = label.to(self.device)

                for optim in self.student_optimizers:
                    optim.zero_grad()

                avg_student_loss = 0
                for i in range(num_students):
                    student_loss = 0
                    for j in range(num_students):
                        if i == j:
                            continue
                        student_loss += self.loss_fn(
                            self.student_cohort[i](data), self.student_cohort[j](data)
                        )
                    student_loss /= num_students - 1
                    student_loss += F.cross_entropy(self.student_cohort[i](data), label)

                    student_loss.backward()
                    self.student_optimizers[i].step()

                    avg_student_loss += student_loss

                avg_student_loss /= num_students

                predictions = []
                correct_preds = []
                for i, student in enumerate(self.student_cohort):
                    predictions.append(student(data).argmax(dim=1, keepdim=True))
                    correct_preds.append(
                        predictions[i].eq(label.view_as(predictions[i])).sum().item()
                    )

                correct += max(correct_preds)

                epoch_loss += avg_student_loss

            for scheduler in self.lr_schedulers:
                scheduler.step()

            epoch_acc = correct / length_of_dataset

            if self.log:
                self.writer.add_scalar("Training loss/Student", epoch_loss, epochs)
                self.writer.add_scalar("Training accuracy/Student", epoch_acc, epochs)

            loss_arr.append(epoch_loss)
            print(f"Epoch: {ep + 1}, Loss: {epoch_loss}, Training Accuracy: {epoch_acc}")

        if plot_losses:
            plt.plot(loss_arr)
        # return lr

    def _evaluate_model(self, model, verbose=True):
        """
        Evaluate the given model's accuaracy over val set.
        For internal use only.

        :param model (nn.Module): Model to be used for evaluation
        :param verbose (bool): Display Accuracy
        """
        model.eval()
        length_of_dataset = len(self.val_loader.dataset)
        correct = 0
        outputs = []

        with torch.no_grad():
            for data, target in self.val_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = model(data)

                if isinstance(output, tuple):
                    output = output[0]
                outputs.append(output)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        if verbose:
            print(f"Accuracy: {correct / length_of_dataset}")

        epoch_val_acc = correct / length_of_dataset
        return outputs, epoch_val_acc

    def evaluate(self):
        """
        Evaluate method for printing accuracies of the trained student networks

        """

        for i, student in enumerate(self.student_cohort):
            print("-" * 80)
            model = deepcopy(student).to(self.device)
            print(f"Evaluating student {i}")
            _, _ = self._evaluate_model(model)

    def get_parameters(self):
        """
        Get the number of parameters for the teacher and the student network
        """

        print("-" * 80)
        for i, student in enumerate(self.student_cohort):
            student_params = sum(p.numel() for p in student.parameters())
            print(f"Total parameters for the student network {i} are: {student_params}")
