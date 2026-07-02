
def read_test_image(image_index, test_loader):
  for idx, (img, out, path) in enumerate(test_loader):
    if idx == image_index:
      data_path = path
      image = img
      label = out

  print(data_path)
  print(image.shape)
  print(label.shape)

  return image, label