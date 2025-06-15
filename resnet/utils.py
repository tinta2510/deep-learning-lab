import torch

def train(model, train_loader, criterion, optimizer, epoch, device, val_loader=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / total
    train_acc = 100. * correct / total

    print(f"[Train] Epoch {epoch} | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")

    # Optional: evaluate on validation set
    val_loss, val_acc = None, None
    if val_loader:
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, epoch, tag="Val")

    return train_loss, train_acc, val_loss, val_acc

def evaluate(model, loader, criterion, device, epoch=None, tag="Test"):
    model.eval()
    loss_total = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss_total += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = loss_total / total
    acc = 100. * correct / total

    if epoch is not None:
        print(f"[{tag}] Epoch {epoch} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")
    else:
        print(f"[{tag}] | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

    return avg_loss, acc