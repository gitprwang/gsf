import torch
import math
import os
import time
import copy
import numpy as np
from lib.logger import get_logger
from lib.metrics import All_Metrics
class Trainer(object):
    def __init__(self, model, loss, optimizer, train_loader, val_loader, test_loader,
                 scaler, args, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        if val_loader != None:
            self.val_per_epoch = len(val_loader)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')
        #log
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        #if not args.debug:
        #self.logger.info("Argument: %r", args)
        # for arg, value in sorted(vars(args).items()):
        #     self.logger.info("Argument %s: %r", arg, value)

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0
        loss_func = torch.nn.L1Loss().to(self.args.device)
        with torch.no_grad():
            b=time.time()
            for batch_idx, (data, target) in enumerate(val_dataloader):
                data = data[..., :self.args.input_dim]
                label = target[..., :self.args.output_dim]
                output = self.model(data)
                if self.args.real_value:
                    label = self.scaler.inverse_transform(label)
                    output = output.cuda()
                    # label = label + torch.randn_like(label)*10
                else:
                    label = self.scaler.inverse_transform(label)
                    output = self.scaler.inverse_transform(output.cuda())
                loss = loss_func(output, label)
                #a whole batch of Metr_LA is filtered
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
            print('val time cost:', time.time()-b)
        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        return val_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        b = time.time()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data[..., :self.args.input_dim]
            label = target[..., :self.args.output_dim]  # (..., 1)
            self.optimizer.zero_grad()


            #data and target shape: B, T, N, F; output shape: B, T, N, F
            output = self.model(data)

            if self.args.real_value:
                label = self.scaler.inverse_transform(label)
                if self.args.use_trick: 
                    label = label + ((101-epoch%100)/100)*torch.randn_like(label)*10
            loss = self.loss(output.cuda(), label)
            loss.backward()

            # add max grad clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()

            #log information
            if batch_idx % self.args.log_step == 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                    epoch, batch_idx, self.train_per_epoch, loss.item()))
        train_epoch_loss = total_loss/self.train_per_epoch
        print('train epoch time cost : {}'.format(time.time()-b))
        self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f}, tf_ratio: {:.6f}'.format(epoch, train_epoch_loss, 0))

        #learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()
        return train_epoch_loss

    def train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        for epoch in range(1, self.args.epochs + 1):
            #epoch_time = time.time()
            train_epoch_loss = self.train_epoch(epoch)
            #print(time.time()-epoch_time)
            #exit()
            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader)

            #print('LR:', self.optimizer.param_groups[0]['lr'])
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            #if self.val_loader == None:
            #val_epoch_loss = train_epoch_loss
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.args.early_stop_patience))
                    break
            # save the best state
            if best_state == True:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())
                torch.save(self.model, self.best_path)
                self.logger.info("Saving current best --- whole --- model to " + self.best_path)

        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))


        #test
        self.model.load_state_dict(best_model)
        self.test(self.model, self.args, self.test_loader, self.scaler, self.logger)

        if self.args.finetune:
            self.finetune(best_loss, best_model)
        #self.val_epoch(self.args.epochs, self.test_loader)
        

    def finetune(self, best_loss, best_model):
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()

        # init optimizer
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.args.lr_init/self.args.finetune_scale, eps=1.0e-8,
                             weight_decay=self.args.weight_decay_rate, amsgrad=False)

        for epoch in range(1, self.args.epochs + 1):
            #epoch_time = time.time()
            train_epoch_loss = self.train_epoch(epoch)
            #print(time.time()-epoch_time)
            #exit()
            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader)

            #print('LR:', self.optimizer.param_groups[0]['lr'])
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            #if self.val_loader == None:
            #val_epoch_loss = train_epoch_loss
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.args.early_stop_patience))
                    break
            # save the best state
            if best_state == True:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())
                torch.save(self.model, self.best_path)
                self.logger.info("Saving current best --- whole --- model to " + self.best_path)

        training_time = time.time() - start_time
        self.logger.info("Total finetune time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))
        self.model.load_state_dict(best_model)
        self.test(self.model, self.args, self.test_loader, self.scaler, self.logger)


    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    @staticmethod
    def test(model, args, data_loader, scaler, logger, path=None):
        if path != None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        y_pred = []
        y_true = []
        time_cost = 0
        with torch.no_grad():
            b = time.time()
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data[..., :args.input_dim]
                label = target[..., :args.output_dim]

                # 给data加入噪声
                # print('add noise into testing data.')
                # data = data*1.1

                output = model(data)
                y_true.append(label)
                y_pred.append(output)
            time_cost = time.time()-b
            print('testing time cost is :', time_cost)
        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        if args.real_value:
            y_pred = torch.cat(y_pred, dim=0)
        else:
            y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
        np.save('./{}_true.npy'.format(args.dataset), y_true.cpu().numpy())
        np.save('./{}_pred.npy'.format(args.dataset), y_pred.cpu().numpy())
        for t in range(y_true.shape[1]):
            mae, rmse, mape, _, _ = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
                                                args.mae_thresh, args.mape_thresh)
            logger.info("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
                t + 1, mae, rmse, mape*100))
        mae, rmse, mape, _, _ = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        logger.info("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
                    mae, rmse, mape*100))
        return time_cost

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return k / (k + math.exp(global_step / k))