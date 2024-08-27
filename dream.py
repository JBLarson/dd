import tensorflow as tf
from PIL import Image
import numpy as np

def genDream(imageIn, layerList, stepSize=100, iterations=10):
	# Load and preprocess the image
	img = Image.open(imageIn)
	img = np.array(img)  # Convert to array
	if img.shape[-1] == 4:  # Check if the image has an alpha channel
		img = img[..., :3]  # Remove alpha channel if present
	img = img.astype('float32')  # Convert image to float32
	img = tf.keras.applications.inception_v3.preprocess_input(img[tf.newaxis,...])

	# Load the InceptionV3 model without the top layer and with ImageNet weights
	model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

	# Define the loss calculation with proper tensor handling
	def calc_loss(img, model):

		# Get the output of these layers
		layer_outputs = [model.get_layer(name).output for name in layerList]
		# Create a new model that returns these outputs
		dream_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
		# Get the activations for the input image
		activations = dream_model(img)
		
		# Calculate the loss as the mean of each activation, then sum them up
		losses = [tf.reduce_mean(act) for act in activations]  # Calculate mean for each specified layer
		total_loss = tf.reduce_sum(losses)  # Sum all the mean values to get a scalar loss
		return total_loss


	# Gradient ascent function to update the image with respect to calculated losses
	@tf.function
	def deep_dream_step(img, model, stepSize):
		with tf.GradientTape() as tape:
			tape.watch(img)
			loss = calc_loss(img, model)
		gradients = tape.gradient(loss, img)
		gradients /= (tf.norm(gradients) + 1e-8)

		#tf.print("Gradient norm:", tf.norm(gradients))  # debugging

		img += gradients * stepSize
		img = tf.clip_by_value(img, -1, 1)  # Keep image in the typical preprocessing range
		return img, loss

	# debugging stat to verify image was preprocessed sucessfully
	#print("Initial image stats:", tf.reduce_min(img).numpy(), tf.reduce_max(img).numpy())

	for i in range(iterations):
		img, loss = deep_dream_step(img, model, stepSize)
		print(f"Iteration {i+1}/{iterations} complete. Loss: {round(loss.numpy(), 6)}")

	# Convert the processed tensor back to an image
	final_img = (img.numpy().squeeze() + 1) * 127.5  # Rescale back to 0-255 range
	final_img = np.clip(final_img, 0, 255)  # Clip to valid pixel range
	final_img = Image.fromarray(np.uint8(final_img))  # Convert to PIL Image for display

	layerString = 'm'
	for layer in layerList:
		layerString += layer.split("mixed")[1]
	outString = imageIn.split('.')[0] + "/" + layerString + "_s" + str(stepSize) + "_i" + str(iterations) + ".png"

	final_img.save(outString)



# genDream(imageIn, layerList, stepSize, iterations)

#gd0 = genDream('starryNight.jpeg', ['mixed1', 'mixed3', 'mixed6'], 42, 7)

gd0 = genDream('starryNight.jpeg', ['mixed3', 'mixed6'], 42, 7)
gd1 = genDream('starryNight.jpeg', ['mixed1', 'mixed6'], 42, 7)
gd2 = genDream('starryNight.jpeg', ['mixed1', 'mixed3'], 42, 7)

