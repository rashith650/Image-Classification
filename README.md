
# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

Include the Problem Statement and Dataset.

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:
Write your own steps

### STEP 2:

### STEP 3:


## PROGRAM

### Name: MOHAMED RASHITH S
### Register Number: 212223243003
```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 7 * 7, 128) # Calculate the input size after pooling
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10) # 10 output classes for MNIST

    def forward(self, x):
        # Forward pass through convolutional layers
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        # Flatten the output for fully connected layers
        x = x.view(-1, 32 * 7 * 7) # Reshape the tensor

        # Forward pass through fully connected layers
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)

        return x

```

```python
# Initialize model, loss function, and optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

```python
# Train the Model
def train_model(model, train_loader, num_epochs=3):
    model.train() # Set the model to training mode

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            if torch.cuda.is_available():
                device = torch.device("cuda")
                images = images.to(device)
                labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('Name: MOHAMED RASHITH S')
        print('Register Number: 212223243003')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
```

## OUTPUT
### Training Loss per Epoch

<img width="343" height="90" alt="image" src="https://github.com/user-attachments/assets/de06b903-f27f-43ea-8e9d-da05ba167639" />


### Confusion Matrix


<img width="820" height="561" alt="image" src="https://github.com/user-attachments/assets/2bcabb66-666b-4ff9-8a79-02fab7821f1b" />



### Classification Report
<img width="671" height="350" alt="image" src="https://github.com/user-attachments/assets/0f24ac56-b391-4326-a9a0-24e63bc9150c" />




### New Sample Data Prediction


<img width="750" height="460" alt="image" src="https://github.com/user-attachments/assets/d8ec9a4c-fcac-49c3-8cf5-6e769be8059f" />


## RESULT
The Convolutional Neural Network was successfully implemented for FashionMNIST image classification. The model achieved good accuracy on the test dataset and produced reliable predictions for new images, proving its effectiveness in extracting spatial features from images.
