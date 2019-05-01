from cam import *
from models import *
from utils import *

# TODO: write a pipeline for loading the training/validation data to be loaded into a certain format and passed as input

NUMBER_EPOCHS = 3
model = Gw00()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    model.cuda()
else:
    print("CUDA unavailable, using CPU!")

"""
for epoch in range(NUMBER_EPOCHS):
    training_data = batch_generator(train_data, batch_size=BATCH_SIZE)
    validation_data = create_batches(val_data, batch_size=1)
    losses = []
    model.train()
    print("Epoch:\t", epoch, "\tof\t", NUMBER_EPOCHS)
    for i_batch, batch in enumerate(training_data):
        inputs = batch['input']
        outputs = model(inputs)
        loss = criterion(outputs, batch['target'])
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().detach().numpy())

    writer.add_scalar('Training Loss', float(np.mean(losses)), epoch)
    json_saver['train_loss'][str(epoch)] = float(np.mean(losses))

    with torch.no_grad():
        val_loss = 0
        for i_batch, batch in enumerate(validation_data):
            model.eval()
            inputs = batch['input']
            outputs = model(inputs)
            outputs = outputs[0].cpu().view(VAL_IMAGE_SIZE).detach().numpy()
            ground = batch['target'].cpu().view(VAL_IMAGE_SIZE).detach().numpy()
            outputs = np.asarray([[0. if pixel < THRESHOLD else 1. for pixel in row] for row in outputs])
            diff = outputs - ground
            squared = np.square(diff)
            accuracy = np.sum(squared) / diff.size
            val_loss += accuracy

    val_loss /= len(validation_data)
    writer.add_scalar('Validation Loss', float(val_loss), epoch)
    json_saver['val_loss'][str(epoch)] = float(val_loss)

    if val_loss < best_val:
        torch.save(model.state_dict(), model_dir + '/model.pt')
        json_saver['model_save_epoch'] = epoch
        best_val = val_loss
        best_epoch = epoch

    with open(save_dir + '/data.json', 'w') as fp:
        json.dump(json_saver, fp)

    if epoch - best_epoch > EARLY_STOPPING:
        break

json_saver['end_time'] = int(time.time())
json_saver['run_time'] = json_saver['end_time'] - json_saver['start_time']
print("DONE TRAINING IN " + str(json_saver['run_time']) + "SECONDS")

with open(save_dir + '/data.json', 'w') as fp:
    json.dump(json_saver, fp, indent=2)
print("SAVED TRAINING DATA")

print("STARTING EVALUATION")
model.load_state_dict(torch.load(model_dir + '/model.pt'))
model.cuda()
model.eval()
predictions = evaluate(save_dir, model, THRESHOLD)
"""