from torch.autograd import Variable
from torch import argmax
from numpy import mean
from sklearn.metrics import accuracy_score


def train_model(model, criterion, optimizer, train_dataloader, test_dataloader= None, epochs= 10, iterations= None, accuracy= False):

    if (iterations is None) | (iterations > train_dataloader.__len__()):
        train_iterations = train_dataloader.__len__()
    else:
        train_iterations = iterations

    test_iterations = train_iterations

    if (test_dataloader is not None):
        test_iterations = test_dataloader.__len__() if test_iterations > test_dataloader.__len__() else test_iterations

    train_loss_list = []
    test_loss_list = []

    train_acc_list = []
    test_acc_list = []

    for e in range(epochs):
        train_epoch_loss = []
        train_epoch_acc = []

        dataiter = iter(train_dataloader)

        model.train(True)
        for i in range(train_iterations):
            model.zero_grad()

            inputs, labels = dataiter.next()
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

            batch_size = inputs.shape[0]

            outputs_l, outputs_c = model(inputs)

            loss = criterion(outputs_l, outputs_c, labels)
            loss.backward()
            optimizer.step()

            train_loss_list.append(float(loss))
            train_epoch_loss.append(float(loss))
            if accuracy:
                epoch_acc = accuracy_score(labels[:,2], argmax(outputs_c.view(batch_size, -1), 1))
                train_epoch_acc.append(epoch_acc)
                train_acc_list.append(epoch_acc)


        if test_dataloader is not None:
            test_epoch_loss = []
            test_epoch_acc = []

            dataiter = iter(test_dataloader)

            model.train(False)
            for i in tqdm(range(test_iterations)):
                inputs, labels = dataiter.next()
                inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

                batch_size = inputs.shape[0]
                outputs_l, outputs_c = model(inputs)

                outputs_l, outputs_c = outputs_l.view(batch_size, -1), outputs_c.view(batch_size, -1)

                loss = criterion(outputs_l, outputs_c)

                test_loss_list.append(float(loss))
                test_epoch_loss.append(float(loss))

            print('Epoch {}, loss {}{}'.format(e, mean(test_epoch_loss),
                                               ', accuracy {}'.format(mean(test_epoch_acc)) if accuracy else ''
                                               )
                  )
        else:
            print('Epoch {}, loss {}{}'.format(e, mean(train_epoch_loss),
                                               ', accuracy {}'.format(mean(train_epoch_acc)) if accuracy else ''
                                               )
                  )

    print('Finished training!')

    return {
            'train_loss_list': train_loss_list,
            'test_loss_list': test_loss_list,
            'train_acc_list': train_acc_list,
            'test_acc_list': test_acc_list
            }
