# coding = utf-8
import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import traceback

def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    steps = 0
    best_acc = 0
    last_step = 0
    
    
    for epoch in range(1, args.epochs+1): 
        print('\nEpoch:%s\n'%epoch)
        model.train()
        for batch in train_iter:
            question1, question2, target = batch.question1, batch.question2, batch.label
            if args.cuda:
                question1, question2, target = question1.cuda(), question2.cuda(), target.cuda()
            optimizer.zero_grad()
            logit = model(question1, question2)
            target = target.type(torch.cuda.FloatTensor)
            criterion = nn.MSELoss()
            loss = criterion(logit, target)
            loss.backward()
            optimizer.step()
            
            

            steps += 1
            if steps % args.log_interval == 0:
                corrects = 0 
                length = len(target.data)
                for i in range(length):
                    a = logit[i].data
                    b = target[i].data
                    if a < 0.5 and b == 0:
                        corrects += 1
                    elif a >= 0.5 and b == 1:
                        corrects += 1
                    else:
                        pass
                accuracy = 100.0 * corrects/batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - acc: {:.4f}%({}/{})'.format(steps, 
                                                                accuracy,
                                                                corrects,
                                                                batch.batch_size))
            if steps % args.test_interval == 0:
                dev_acc = eval(dev_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps, best_acc)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))



def eval(data_iter, model, args):
    model.eval()
    corrects = 0
    for batch in data_iter:
        question1, question2, target = batch.question1, batch.question2, batch.label
        if args.cuda:
            question1, question2, target = question1.cuda(), question2.cuda(), target.cuda()

        logit = model(question1, question2)

        
        target = target.type(torch.cuda.FloatTensor)
        # target = target.type(torch.FloatTensor)

        length = len(target.data)
        for i in range(length):
            a = logit[i].data.cpu().numpy()
            b = target[i].data.cpu().numpy()
            print('%s,   %s' %(str(a), str(b)))
            if a < 0.5 and b == 0:
                corrects += 1
            elif a >= 0.5 and b == 1:
                corrects += 1
            else:
                pass
        
    size = float(len(data_iter.dataset))
    accuracy = 100.0 * float(corrects)/size
    print('\nEvaluation -  acc: {:.4f}%({}/{}) \n'.format(accuracy, 
                                                          corrects, 
                                                          size))
    return accuracy


def test(test_iter, model, args):
    threshold = 0.5
    res = []
    for batch in test_iter:
        qid, question1, question2 = batch.pid, batch.question1, batch.question2
        # if args.cuda:
        #     qid, question1, question2 = qid.cuda(), question1.cuda(), question2.cuda()
        results = model(question1, question2)
        for i in range(len(qid.data)):
            if results[i].data >= threshold:
                res.append([qid[i].data.cpu().numpy(), '1'])
            #elif results.data[i] < threshold:
            else:
                res.append([qid[i].data.cpu().numpy(), '0'])
    
    # res = sorted(res, key=lambda x: x[0])
    with open(args.res_path, 'w') as f:
        cnt = 1
        for x in res:
            f.write('{}\t{}\n'.format(x[0], x[1]))
            cnt += 1
    
    with open(args.res_path, 'r') as fin:
        for line in fin:
            lineno, label = line.strip().split('\t')
            lineno = int(lineno)
    

    # res = pd.DataFrame(res, columns=['id', 'label'])
    # res.to_csv(args.res_path, sep='\t', index=False, header=None)


def save(model, save_dir, save_prefix, steps, acc):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}_{}.pt'.format(save_prefix, steps, acc)
    torch.save(model.state_dict(), save_path)

