import simplified_facenet as facenet

# lfw_dir = "lfw_aligned"
# file_ext = "png"
lfw_dir = "lfw"
file_ext = "jpg"

# Ronaldo_Luis_Nazario_de_Lima 1 2
# Junichiro_Koizumi 49 George_W_Bush 5

image_size = 160
paths_batch = [
	"C:/facenetTestFolder/" + lfw_dir + "/Ronaldo_Luis_Nazario_de_Lima/Ronaldo_Luis_Nazario_de_Lima_0001." + file_ext,
	"C:/facenetTestFolder/" + lfw_dir + "/Ronaldo_Luis_Nazario_de_Lima/Ronaldo_Luis_Nazario_de_Lima_0002." + file_ext,
	"C:/facenetTestFolder/" + lfw_dir + "/Junichiro_Koizumi/Junichiro_Koizumi_0049." + file_ext,
	"C:/facenetTestFolder/" + lfw_dir + "/George_W_Bush/George_W_Bush_0005." + file_ext
]
images = facenet.load_data(paths_batch, False, False, image_size)