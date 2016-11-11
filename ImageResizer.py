from PIL import Image, ImageOps
from os import listdir, makedirs
from os.path import isfile, join, basename, isdir

def resize_image(source, destination):
    img = Image.open(source)
    img = img.convert('RGB')
    img = ImageOps.fit(img, (224, 224), Image.ANTIALIAS)
    img.save(destination, "JPEG", quality=80, optimize=True, progressive=True)



def resize_categorie(source, destination):
    all_files = [f for f in listdir(source) if isfile(join(source, f))]
    for file in all_files:
        if file.endswith(".npy"):
            continue
        path = join(source, file)
        new_name = basename(file)
        destination_file = join(destination, new_name)
        resize_image(path, destination_file)

def resize_dataset(source, destination):
    all_dirs = [f for f in listdir(source) if isdir(join(source, f))]
    for file in all_dirs:
        if file.endswith(".npy"):
            continue
        path = join(source, file)
        new_name = basename(file)
        destination_folder = join(destination, new_name)
        print(destination_folder)
        makedirs(destination_folder)
        resize_categorie(path, destination_folder)


