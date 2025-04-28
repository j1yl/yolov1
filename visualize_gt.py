from dataset import VOCDataset, visualize_voc_batch
voc = VOCDataset("data/VOC2012", split="train", transform=None, S=7, B=2, C=20)
visualize_voc_batch(voc, batch_size=4, class_names=voc.class_names)