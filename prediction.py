import io
import os
import scipy.misc
import numpy as np
import six
import time
import glob
from IPython.display import display
from six import BytesIO
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from PIL import Image
import cv2
from matplotlib import pyplot as plt

def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: a file path (this can be local or on colossus)

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

train_record_path = 'train.record' ##path of google colab
test_record_path = 'test.record'
labelmap_path = 'labelmap.pbtxt'
output_directory = 'inference_graph/'
pipeline_config_path = 'efficientdet_d0_coco17_tpu-32/pipeline.config'  ## edit the file 1st
model_dir = 'training/' ## checkpoints to be stored inside of this folder

category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)



tf.keras.backend.clear_session()
model = tf.saved_model.load(f'/content/{output_directory}/saved_model')

batch_size = 16
num_steps = 20000
num_eval_steps = 1000

def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict

def get_bounding_boxes_cooper(image_path,output_dict):
  cords = []
  im = Image.open(image_path)
  width, height = im.size

  # This is the way I'm getting my coordinates
  boxes = output_dict['detection_boxes']
  # get all boxes from an array
  max_boxes_to_draw = boxes.shape[0]
  # get scores to get a threshold
  scores = output_dict['detection_scores']
  # this is set as a default but feel free to adjust it to your needs
  min_score_thresh=.5
  # iterate over all objects found
  for i in range(min(max_boxes_to_draw, boxes.shape[0])):
      if scores is None or scores[i] > min_score_thresh:
          
          class_name = category_index[output_dict['detection_classes'][i]]['name']
          box = boxes[i]
          xmin = box[1]
          ymin = box[0]
          xmax = box[3]
          ymax = box[2]
          (left, right, top, bottom) = (xmin * width, xmax * width, ymin * height, ymax * height)
          bbox = [int(left), int(right), int(top), int(bottom)]
          cords.append({"bbox":bbox,"class_name":class_name,"label":output_dict['detection_classes'][i],"score":scores[i]})

  return cords


for image_path in glob.glob('images/test/*.jpg'):
    image_np = load_image_into_numpy_array(image_path)
    output_dict = run_inference_for_single_image(model, image_np)
    ans = get_bounding_boxes_cooper(image_path,output_dict)
    print(ans)

    for i in range(0,len(ans),1):
      x1 = ans[i]["bbox"][0]
      x3 = ans[i]["bbox"][1]
      y1 = ans[i]["bbox"][2]
      y3 = ans[i]["bbox"][3]
      rectangle = cv2.rectangle(image,(x1,y1),(x3,y3),(0,255,0),3)

    plt.imshow(rectangle)
    plt.show()




    

