# Pályaszínhez adaptívan alkalmazkodó turtlebot vonalkövetése neurális hálózattal ROS-környezetben
Multiple colored line following with neural network using ROS and turtlebot
## Tartalomjegyzék
1. [Előkészületek](##1.-előkészületek)  
2. [Első lépések](##Első-lépések)  
3. [Feladat kidolgozása és megoldása](##3.-Feladat-kidolgozása-és-megoldása)
4. [Eredmények](##4.-Eredmények)  
## 1. Előkészületek
- A projekt első lépéseként telepítettük az `Ubuntu 20.04`-verziószámú operációs rendszert `Oracle VM VirtualBox` szoftver segítségével virtuális környezetben.
> ![kép](https://github.com/4isSorin/Line_Following_kogrob/assets/167373493/5ffaaaae-4031-40e2-b34c-9afc5df87947)
> > Alapadatok a rendszerről

Ezután a friss felületre telepítettük a feladat megoldásához szükséges szoftvereket (mint például: `ROS Noetic` - `full desktop`csomag, `Python3.0`, `PyCharm Community Edition`). Majd elkezdtük letölteni a várhatóan szükséges Python könyvtárakat, melyek számát a feladat megoldása során szükség szerint tovább bővítettük. Az említett könyvtárak pontos nevei a dokumentációhoz csatolt kódok importálási szekciójában lesznek láthatóak.
## 2. Első lépések
### Projekt feladatának meghatározása
- Mielőtt továbbléptünk volna, tisztáznunk kellett, hogy pontosan milyen célt szeretnénk elérni a projekt során a vonalkövetés témakörön belül. A választás arra esett, hogy a robotnak képesnek kell lennie neurális hálót használva többszínű vonallal ellátott pályát követni úgy, hogy meg is különböztesse a tanított színeket és különböző színek esetén különböző sebességgel haladjon.  
### Pálya elkészítése
- A vonalkövetéshez elengedhetetlen volt egy saját pálya, melyen betaníthatjuk és tesztelhetjük a robotot. Ezt a pályát `Blender 4.0` segítségével készítettük el.
> ![kép](https://github.com/4isSorin/Line_Following_kogrob/assets/167373493/bc0be3e7-94a3-49f5-baa3-ff3c2224ecf9)
> > Kész pálya három különböző színnel
- A pályaát collada (.dae) formátumba exportáltuk ki, ugyanis ez kedvező lesz a továbbiakban a Gazebo-ba való importálás szempontjából.
- A `gazebo` parancs segítségével megnyílik a környezet, majd az `Edit` menüpont alatt kiválasztjuk a `Model Editor`-t, melyen belül a `Custom Shapes` `Add` gombjára kattintva betallózzunk a létrehozott fájlt. A `Link Inspector` menüt előhozva célszerű a `Kinematic` pontot True értékre állítani, majd a modellt elmenteni. A pályán látható színek egyenlőre nem egyeznek meg a Blender-ben létrehozottakkal, azonban ez kiküszöbölhető a `model.sdf` fájljának módosításával a következőképpen: a `visual` szekcióban a `material` részhez tartozó sorokat kitöröljük. Ezután már a Gazebo környezetében is pontosan látszanak a beállított színek.
## Kamera hozzáadása a robothoz
A kamera hozzáadásához két fájl módosítása szükséges. Először a robot 3D modelljéhez adjuk hozzá, mely egy 45°-ban döntött piros kockaként jelenik meg. Ehhez a `/turtlebot3/turtlebot3_description/urdf/turtlebot3_burger.urdf.xacro` fájlba az alábbi sorok implementálása szükséges:
```xml
...
  <!-- Camera -->
  <joint type="fixed" name="camera_joint">
    <origin xyz="0.03 0 0.11" rpy="0 0.79 0"/>
    <child link="camera_link"/>
    <parent link="base_link"/>
    <axis xyz="0 1 0" />
  </joint>

  <link name='camera_link'>
    <pose>0 0 0 0 0 0</pose>
    <inertial>
      <mass value="0.001"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia
          ixx="1e-6" ixy="0" ixz="0"
          iyy="1e-6" iyz="0"
          izz="1e-6"
      />
    </inertial>

    <collision name='collision'>
      <origin xyz="0 0 0" rpy="0 0 0"/> 
      <geometry>
        <box size=".01 .01 .01"/>
      </geometry>
    </collision>

    <visual name='camera_link_visual'>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size=".02 .02 .02"/>
      </geometry>
    </visual>

  </link>

  <gazebo reference="camera_link">
    <material>Gazebo/Red</material>
  </gazebo>

  <joint type="fixed" name="camera_optical_joint">
    <origin xyz="0 0 0" rpy="-1.5707 0 -1.5707"/>
    <child link="camera_link_optical"/>
    <parent link="camera_link"/>
  </joint>

  <link name="camera_link_optical">
  </link>
...
```

Ezután a `/turtlebot3/turtlebot3_description/urdf/turtlebot3_burger.gazebo.xacro` fájl kiegészítése szükséges a lenti sorokkal. Ez hozza létre a szimulált kamerát.
```xml
...
  <!-- Camera -->
  <gazebo reference="camera_link">
    <sensor type="camera" name="camera">
      <update_rate>30.0</update_rate>
      <visualize>false</visualize>
      <camera name="head">
        <horizontal_fov>1.3962634</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>10.0</far>
        </clip>
        <noise>
          <type>gaussian</type>
          <!-- Noise is sampled independently per pixel on each frame.
               That pixel's noise value is added to each of its color
               channels, which at that point lie in the range [0,1]. -->
          <mean>0.0</mean>
          <stddev>0.007</stddev>
        </noise>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <alwaysOn>true</alwaysOn>
        <updateRate>0.0</updateRate>
        <cameraName>camera</cameraName>
        <imageTopicName>image</imageTopicName>
        <cameraInfoTopicName>camera_info</cameraInfoTopicName>
        <frameName>camera_link_optical</frameName>
        <hackBaseline>0.0</hackBaseline>
        <distortionK1>0.0</distortionK1>
        <distortionK2>0.0</distortionK2>
        <distortionK3>0.0</distortionK3>
        <distortionT1>0.0</distortionT1>
        <distortionT2>0.0</distortionT2>
      </plugin>
    </sensor>
  </gazebo>
...
```
## A tanításhoz használandó képek elkészítése
A jól kondícionált tudás eléréséhez nagyméretű tanító adathatlmazra lesz szükség. Ha minden különböző vonalszínt és lehetséges trajektóriát szeretnénk reprezentálni változatos orientációkból, ahhoz a legkézenfekvőbb megoldás a pálya teljes bejárása közbeni fényképes dokumentálás. A robot kézi manóverezéséhez a távirányító node használható, mely a `W,A,S,D,X` billentyűk általi irányítást teszi lehetővé.
```console
roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
```
Ezzel párhuzamosan egy másik terminálban elindítható a `save_training_images.py`, mely 320x240 felbontású képeket ment a `Space` billentyű lenyomása esetén.
```python
#!/usr/bin/env python3

import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage
import rospy
import rospkg
try:
    from queue import Queue
except ImportError:
    from Queue import Queue
import threading
from datetime import datetime

class BufferQueue(Queue):
    """Slight modification of the standard Queue that discards the oldest item
    when adding an item and the queue is full.
    """
    def put(self, item, *args, **kwargs):
        # The base implementation, for reference:
        # https://github.com/python/cpython/blob/2.7/Lib/Queue.py#L107
        # https://github.com/python/cpython/blob/3.8/Lib/queue.py#L121
        with self.mutex:
            if self.maxsize > 0 and self._qsize() == self.maxsize:
                self._get()
            self._put(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()


class cvThread(threading.Thread):
    """
    Thread that displays and processes the current image
    It is its own thread so that all display can be done
    in one thread to overcome imshow limitations and
    https://github.com/ros-perception/image_pipeline/issues/85
    """
    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.queue = queue
        self.image = None
        

    def run(self):
        if withDisplay:
            cv2.namedWindow("display", cv2.WINDOW_NORMAL)
                
        while True:
            self.image = self.queue.get()

            processedImage = self.processImage(self.image) # only resize

            if withDisplay:
                cv2.imshow("display", processedImage)
                
            k = cv2.waitKey(6) & 0xFF
            if k in [27, ord('q')]: # 27 = ESC
                rospy.signal_shutdown('Quit')
            elif k in [32, ord('s')]: # 32 = Space
                time_prefix = datetime.today().strftime('%Y%m%d-%H%M%S-%f')
                file_name = save_path + time_prefix + ".jpg"
                cv2.imwrite(file_name, processedImage)
                print("File saved: %s" % file_name)

    def processImage(self, img):

        height, width = img.shape[:2]

        if height != 240 or width != 320:
            dim = (320, 240)
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        
        return img


def queueMonocular(msg):
    try:
        # Convert your ROS Image message to OpenCV2
        #cv2Img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8") # in case of non-compressed image stream only
        cv2Img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
    except CvBridgeError as e:
        print(e)
    else:
        qMono.put(cv2Img)

print("OpenCV version: %s" % cv2.__version__)

queueSize = 1      
qMono = BufferQueue(queueSize)

bridge = CvBridge()
    
rospy.init_node('image_listener')

withDisplay = bool(int(rospy.get_param('~with_display', 1)))
rospack = rospkg.RosPack()
path = rospack.get_path('turtlebot3_mogi')
save_path = path + "/saved_images/"
print("Saving files to: %s" % save_path)

# Define your image topic
image_topic = "/camera/image/compressed"
# Set up your subscriber and define its callback
rospy.Subscriber(image_topic, CompressedImage, queueMonocular)

# Start image processing thread
cvThreadHandle = cvThread(qMono)
cvThreadHandle.setDaemon(True)
cvThreadHandle.start()

# Spin until ctrl + c
rospy.spin()
```
[tanításhoz készített képek beszúrása]
Az elkészített felvételeket színtől és cselekvésti tervtől függően 10 mappába soroltuk, melyek a következőek:
-

## A neurális háló létrehozása
A hálózatunk egy LeNet-5 típusú konvulúciós neurális háló, melynek bemenetei 24x24 pixeles RGB-képek. A tanítás a `train_network.py` szkript segítségével törénik:
```python
# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import __version__ as keras_version
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.random import set_seed
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from imutils import paths
import numpy as np
import random
import cv2
import os
import matplotlib.pyplot as plt
from numpy.random import seed

# Set image size
image_size = 24

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Fix every random seed to make the training reproducible
seed(1)
set_seed(2)
random.seed(42)

print("[INFO] Version:")
print("Tensorflow version: %s" % tf.__version__)
keras_version = str(keras_version).encode('utf8')
print("Keras version: %s" % keras_version)

def build_LeNet(width, height, depth, classes):
    # initialize the model
    model = Sequential()
    inputShape = (height, width, depth)

    # first set of CONV => RELU => POOL layers
    model.add(Conv2D(20, (5, 5), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second set of CONV => RELU => POOL layers
    model.add(Conv2D(50, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))

    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    # return the constructed network architecture
    return model

    
dataset = '..//training_images'
# initialize the data and labels
print("[INFO] loading images and labels...")
data = []
labels = []
 
# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(dataset)))
random.shuffle(imagePaths)
# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (image_size, image_size))
    image = img_to_array(image)
    data.append(image)
    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    print("Image: %s, Label: %s" % (imagePath, label))
    if label == 'forward':
        label = 0
    elif label == 'right':
        label = 1
    elif label == 'left':
        label = 2
    else:
        label = 3
    labels.append(label)
    
    
# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
 
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=4)
testY = to_categorical(testY, num_classes=4)


# initialize the number of epochs to train for, initial learning rate,
# and batch size
EPOCHS  = 40
INIT_LR = 0.001
DECAY   = INIT_LR / EPOCHS
BS      = 32

# initialize the model
print("[INFO] compiling model...")
model = build_LeNet(width=image_size, height=image_size, depth=3, classes=4)
opt = Adam(learning_rate=INIT_LR, decay=DECAY)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
 
# print model summary
model.summary()

# checkpoint the best model
checkpoint_filepath = "..//network_model//model.best.h5"
checkpoint = ModelCheckpoint(checkpoint_filepath, monitor = 'val_loss', verbose=1, save_best_only=True, mode='min')

# set a learning rate annealer
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.5, min_lr=1e-6)

# callbacks
callbacks_list=[reduce_lr, checkpoint]

# train the network
print("[INFO] training network...")
history = model.fit(trainX, trainY, batch_size=BS, validation_data=(testX, testY), epochs=EPOCHS, callbacks=callbacks_list, verbose=1)
 
# save the model to disk
print("[INFO] serializing network...")
model.save("..//network_model//model.h5")

plt.xlabel('Epoch Number')
plt.ylabel("Loss / Accuracy Magnitude")
plt.plot(history.history['loss'], label="loss")
plt.plot(history.history['accuracy'], label="acc")
plt.plot(history.history['val_loss'], label="val_loss")
plt.plot(history.history['val_accuracy'], label="val_acc")
plt.legend()
plt.savefig('model_training')
plt.show()
```
A tanítási folyamat eredményét az epochszám-pontosság grafikon ábrázolja.
[kép a grafikonunkról]

- nano .bashrc
- Elkészítettük a catkin workspace-ünket, ami az egyszerűség kedvéért a `bme_catkin_ws` nevet kapta.
## 3. Feladat megoldása
- Írjunk ilyen "érdekességek" részt a hibákról amik felemrültek? (mondjuk valszeg ez felesleges)
- Feladat megoldásának menetét leírni, csak konkrétan hogy milyen scripteket használtunk, melyik mit csinál, elmesélni a betanítást + pár kép és kész
## 4. Eredmények
[Youtube videó link](https://www.youtube.com/watch?v=jogtECytDSQ&t=0s)
