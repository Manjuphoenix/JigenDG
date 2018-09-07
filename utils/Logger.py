from time import time

from os.path import join, dirname

from .tf_logger import TFLogger

_log_path = join(dirname(__file__), '../logs')


# high level wrapper for tf_logger.TFLogger
class Logger():
    def __init__(self, args, update_frequency=10):
        self.current_epoch = 0
        self.max_epochs = args.epochs
        self.last_update = time()
        self._clean_epoch_stats()
        self.update_f = update_frequency
        self.losses = {"jigsaw": [], "class": []}
        self.val_acc = {"jigsaw": [], "class": []}
        folder, logname = self.get_name_from_args(args)
        log_path = join(_log_path, folder, logname)
        self.tf_logger = TFLogger(log_path)
        print("Saving to %s" % log_path)
        self.current_iter = 0

    def new_epoch(self, learning_rates):
        self.current_epoch += 1
        self.last_update = time()
        self.lrs = learning_rates
        print("New epoch - lr: %s" % ", ".join([str(lr) for lr in self.lrs]))
        self._clean_epoch_stats()
        for n, v in enumerate(self.lrs):
            self.tf_logger.scalar_summary("aux/lr%d" % n, v, self.current_iter)

    def log(self, it, iters, losses, samples_right, total_samples):
        self.current_iter += 1
        loss_string = ", ".join(["%s : %f" % (k, v) for k, v in losses.items()])
        for k, v in samples_right.items():
            past = self.epoch_stats.get(k, 0.0)
            self.epoch_stats[k] = past + v
        self.total += total_samples
        acc_string = ", ".join(["%s : %f" % (k, v / total_samples) for k, v in samples_right.items()])
        if it % self.update_f == 0:
            print("%d/%d of epoch %d/%d %s - acc %s [bs:%d]" % (it, iters, self.current_epoch, self.max_epochs, loss_string,
                                                                acc_string, total_samples))
            for k, v in losses.items():
                self.losses[k].append(v)
            # update tf log
            for k, v in losses.items(): self.tf_logger.scalar_summary("train/loss_%s" % k, v, self.current_iter)

    def _clean_epoch_stats(self):
        self.epoch_stats = {}
        self.total = 0

    def log_test(self, accuracies):
        print("Accuracies on target: " + ", ".join(["%s : %f" % (k, v) for k, v in accuracies.items()]))
        for k, v in accuracies.items():
            self.val_acc[k].append(v)
        for k, v in accuracies.items(): self.tf_logger.scalar_summary("test/acc_%s" % k, v, self.current_iter)

    @staticmethod
    def get_name_from_args(args):
        folder_name = "%s_to_%s" % ("-".join(sorted(args.source)), args.target)
        name = "eps%d_bs%d_lr%g_class%d_jigClass%d_jigWeight%g_%d" % (args.epochs, args.batch_size, args.learning_rate, args.n_classes,
                                                                      args.jigsaw_n_classes, args.jig_weight, int(time() % 1000))
        return folder_name, name
