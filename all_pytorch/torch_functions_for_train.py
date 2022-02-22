import matplotlib as plt
from torch import optim


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()


def train_net(crit, lr, epochs, my_dataloader,net, labels_1d=False):
    optimizer = optim.Adam(net.parameters(), lr)
    counter = []
    loss_history = []
    iteration_number = 0

    # Iterate throught the epochs
    for epoch in range(epochs):

        # Iterate over batches
        for i, (img0, img1, label) in enumerate(my_dataloader, 0):

            # Send the images and labels to CUDA
            img0, img1, label = img0.cuda().permute(0, 3, 1, 2), img1.cuda().permute(0, 3, 1, 2), label.cuda()

            # Zero the gradients
            optimizer.zero_grad()

            # Pass in the two images into the network and obtain two outputs
            output1, output2 = net(img0, img1)

            # Pass the outputs of the networks and label into the loss function
            if labels_1d:
                label = label.reshape(-1)
            loss_contrastive = crit(output1, output2, label)

            # Calculate the backpropagation
            loss_contrastive.backward()

            # Optimize
            optimizer.step()

            # Every 10 batches print out the loss
            if i % 10 == 0:
                print(f"\rEpoch number {epoch}\n Current loss {loss_contrastive.item()}\n", end="")
                iteration_number += 10

                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())

    show_plot(counter, loss_history)
