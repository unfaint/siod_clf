from tqdm import tqdm
from torch.autograd import Variable
from numpy import mean as np_mean


def train_model(model, train_dataloader, criterion, optimizer, test_dataloader= None, epochs= 10, train_iterations= None):
    if train_iterations is None:
        train_iterations = train_dataloader.__len__()
        if test_dataloader is not None:
            test_iterations = test_dataloader.__len__()

    loss_list = []

    for e in range(epochs):
        epoch_loss = []
        epoch_acc = []

        train_dataiter = iter(train_dataloader)

        model.train(True)
        for i in tqdm(range(train_iterations)):
            model.zero_grad()

            inputs, labels = train_dataiter.next()
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

            batch_size = inputs.shape[0]

            outputs_l, outputs_c = model(inputs)

            outputs_l, outputs_c = outputs_l.view(batch_size, -1), outputs_c.view(batch_size, -1)

            loss = criterion(outputs_l, outputs_c, labels)
            loss.backward()
            optimizer.step()

            loss_list.append(float(loss))
            epoch_loss.append(float(loss))

        print('Epoch {}, loss {}'.format(e, np_mean(epoch_loss)))

    print('Training finished!')
