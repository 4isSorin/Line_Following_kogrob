# Pályaszínhez adaptívan alkalmazkodó turtlebot vonalkövetése neurális hálózattal ROS-környezetben
Multiple colored line following with neural network using ROS and turtlebot

[Projekt fájlait tartalmazó drive link](https://drive.google.com/drive/folders/1LQzCVXgYOipmSOD1s8-OItRQ8tjKVA1g?usp=sharing)
## Tartalomjegyzék
1. [Előkészületek](##1.-előkészületek)    
2. [Pálya és kamera modellezése](##2.-Pálya-és-kamera-modellezése)
3. [Feladat megoldásának menete](##3.-Feladat-megoldásának-menete)
4. [Eredmények](##4.-Eredmények)  

## 1. Előkészületek
- A projekt első lépéseként telepítettük az `Ubuntu 20.04`-verziószámú operációs rendszert `Oracle VM VirtualBox` szoftver segítségével virtuális környezetben.
> ![kép](https://github.com/4isSorin/Line_Following_kogrob/assets/167373493/5ffaaaae-4031-40e2-b34c-9afc5df87947)
> > Alapadatok a rendszerről

Ezután a friss felületre telepítettük a feladat megoldásához szükséges szoftvereket (mint például: `ROS Noetic` - `full desktop`csomag, `Python3.0`, `PyCharm Community Edition`). Majd elkezdtük letölteni a várhatóan szükséges Python könyvtárakat, melyek számát a feladat megoldása során szükség szerint tovább bővítettük. Az említett könyvtárak pontos nevei a dokumentációhoz csatolt kódok importálási szekciójában lesznek láthatóak.
### Projekt feladatának meghatározása
- Mielőtt továbbléptünk volna, tisztáznunk kellett, hogy pontosan milyen célt szeretnénk elérni a projekt során a vonalkövetés témakörön belül. A választás arra esett, hogy a robotnak képesnek kell lennie neurális hálót használva többszínű vonallal ellátott pályát követni úgy, hogy meg is különböztesse a tanított színeket és különböző színek esetén különböző sebességgel haladjon.
- Elkészítettük a catkin workspace-ünket, ami az egyszerűség kedvéért a `bme_catkin_ws` nevet kapta. Az említett könyvtárba helyeztük a továbbiakban a projekthez szükséges fájlokat.
### Automatikus parancsok
- A `.bashrc` egy olyan fájl, ami minden terminál indításakor automatikusan végrehajtódik, tehát nem kell többé maunálisan futtatgatnunk ezeket a parancsokat. Mi az órai anyagban megtalálható fájl módosított verzióját használtuk:
```bash
# ~/.bashrc: executed by bash(1) for non-login shells.
# see /usr/share/doc/bash/examples/startup-files (in the package bash-doc)
# for examples

# If not running interactively, don't do anything
case $- in
    *i*) ;;
      *) return;;
esac

# don't put duplicate lines or lines starting with space in the history.
# See bash(1) for more options
HISTCONTROL=ignoreboth

# append to the history file, don't overwrite it
shopt -s histappend

# for setting history length see HISTSIZE and HISTFILESIZE in bash(1)
HISTSIZE=1000
HISTFILESIZE=2000

# check the window size after each command and, if necessary,
# update the values of LINES and COLUMNS.
shopt -s checkwinsize

# If set, the pattern "**" used in a pathname expansion context will
# match all files and zero or more directories and subdirectories.
#shopt -s globstar

# make less more friendly for non-text input files, see lesspipe(1)
[ -x /usr/bin/lesspipe ] && eval "$(SHELL=/bin/sh lesspipe)"

# set variable identifying the chroot you work in (used in the prompt below)
if [ -z "${debian_chroot:-}" ] && [ -r /etc/debian_chroot ]; then
    debian_chroot=$(cat /etc/debian_chroot)
fi

# set a fancy prompt (non-color, unless we know we "want" color)
case "$TERM" in
    xterm-color|*-256color) color_prompt=yes;;
esac

# uncomment for a colored prompt, if the terminal has the capability; turned
# off by default to not distract the user: the focus in a terminal window
# should be on the output of commands, not on the prompt
#force_color_prompt=yes

if [ -n "$force_color_prompt" ]; then
    if [ -x /usr/bin/tput ] && tput setaf 1 >&/dev/null; then
	# We have color support; assume it's compliant with Ecma-48
	# (ISO/IEC-6429). (Lack of such support is extremely rare, and such
	# a case would tend to support setf rather than setaf.)
	color_prompt=yes
    else
	color_prompt=
    fi
fi

if [ "$color_prompt" = yes ]; then
    PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
else
    PS1='${debian_chroot:+($debian_chroot)}\u@\h:\w\$ '
fi
unset color_prompt force_color_prompt

# If this is an xterm set the title to user@host:dir
case "$TERM" in
xterm*|rxvt*)
    PS1="\[\e]0;${debian_chroot:+($debian_chroot)}\u@\h: \w\a\]$PS1"
    ;;
*)
    ;;
esac

# enable color support of ls and also add handy aliases
if [ -x /usr/bin/dircolors ]; then
    test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
    alias ls='ls --color=auto'
    #alias dir='dir --color=auto'
    #alias vdir='vdir --color=auto'

    alias grep='grep --color=auto'
    alias fgrep='fgrep --color=auto'
    alias egrep='egrep --color=auto'
fi

# colored GCC warnings and errors
#export GCC_COLORS='error=01;31:warning=01;35:note=01;36:caret=01;32:locus=01:quote=01'

# some more ls aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'

# Add an "alert" alias for long running commands.  Use like so:
#   sleep 10; alert
alias alert='notify-send --urgency=low -i "$([ $? = 0 ] && echo terminal || echo error)" "$(history|tail -n1|sed -e '\''s/^\s*[0-9]\+\s*//;s/[;&|]\s*alert$//'\'')"'

# Alias definitions.
# You may want to put all your additions into a separate file like
# ~/.bash_aliases, instead of adding them here directly.
# See /usr/share/doc/bash-doc/examples in the bash-doc package.

if [ -f ~/.bash_aliases ]; then
    . ~/.bash_aliases
fi

# enable programmable completion features (you don't need to enable
# this, if it's already enabled in /etc/bash.bashrc and /etc/profile
# sources /etc/bash.bashrc).
if ! shopt -oq posix; then
  if [ -f /usr/share/bash-completion/bash_completion ]; then
    . /usr/share/bash-completion/bash_completion
  elif [ -f /etc/bash_completion ]; then
    . /etc/bash_completion
  fi
fi


# ----- SAJAT  stuff for kognitiv robotika ---------------
#source /opt/ros/noetic/setup.bash
#-----WORKSPACE=~/catkin_ws/devel/setup.bash
#WORKSPACE=~/catkin_ws/devel/setup.bash
#source $WORKSPACE
#export TURTLEBOT3_MODEL=burger

#export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/bme_catkin_ws/src/Week-1-8-Cognitive-robotics/turtlebot3_mogi/gazebo_models/

# -------------- Github bashrc -----------------
# bash colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;36m'
GRAY='\033[0;37m'
LINEARBLUE='\033[1;34m'
# No Color
NC='\033[0m'

# CUDA stuff
export LD_LIBRARY_PATH=/usr/lib/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/cuda/include:$LD_LIBRARY_PATH
export PATH=$PATH:/usr/local/cuda/bin

# ROS workspace stuff
source /opt/ros/noetic/setup.bash
#WORKSPACE=~/catkin_ws/devel/setup.bash
WORKSPACE=~/bme_catkin_ws/devel/setup.bash
source $WORKSPACE

# Automatic ROS IP config
IP_ADDRESSES=$(hostname -I | sed 's/ *$//g')
IP_ARRAY=($IP_ADDRESSES)
FIRST_IP=${IP_ARRAY[0]}

if [ "$FIRST_IP" != "" ];
then
    true
    #echo "There are IP addresses!"
else
    echo "Warning FIRST_IP var was empty:" $FIRST_IP
    echo "Maybe client is not connected to any network?"
    FIRST_IP=127.0.0.1
fi

export ROS_MASTER_URI=http://$FIRST_IP:11311
export ROS_IP=$FIRST_IP

echo -e "=============== ${YELLOW}NETWORK DETAILS${NC} ================="
echo -e ${GREEN}ACTIVE IP ADDRESSES:${NC}
echo $IP_ADDRESSES | tr " " "\n"
echo -e ${GREEN}SELECTED IP ADDRESS:${NC} $FIRST_IP

echo -e "============== ${YELLOW}ROS NETWORK CONFIG${NC} ==============="
echo export ROS_MASTER_URI=$ROS_MASTER_URI
echo export ROS_IP=$ROS_IP

# Chess simulation Gazebo path
#export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/catkin_ws/src/mogi_chess_ros_framework/mogi_chess_gazebo/gazebo_models/

# Turtlebot 3 stuff
export TURTLEBOT3_MODEL=burger
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/bme_catkin_ws/src/Week-1-8-Cognitive-robotics/turtlebot3_mogi/gazebo_models/
export LDS_MODEL=LDS-01

echo -e "================= ${YELLOW}ROS WORKSPACE${NC} ================="
echo -e ${GREEN}WORKSPACE SOURCED:${NC} $WORKSPACE | rev | cut -d'/' -f3- | rev
echo -e ${GREEN}GAZEBO MODEL PATH:${NC}
echo $GAZEBO_MODEL_PATH | tr ":" "\n" | sed '/./,$!d'
echo "================================================="
```
--- 
## 2. Pálya és kamera modellezése
### Pálya elkészítése
- A vonalkövetéshez elengedhetetlen volt egy saját pálya, melyen betaníthatjuk és tesztelhetjük a robotot. Ezt a pályát `Blender 4.0` segítségével készítettük el.
> ![kép](https://github.com/4isSorin/Line_Following_kogrob/assets/167373493/bc0be3e7-94a3-49f5-baa3-ff3c2224ecf9)
> > Kész pálya három különböző színnel
- A pályát collada (.dae) formátumba exportáltuk ki, ugyanis ez kedvező lesz a továbbiakban a Gazebo-ba való importálás szempontjából.
- A `gazebo` parancs segítségével megnyílik a környezet, majd az `Edit` menüpont alatt kiválasztjuk a `Model Editor`-t, melyen belül a `Custom Shapes` `Add` gombjára kattintva betallózzunk a létrehozott fájlt. A `Link Inspector` menüt előhozva célszerű a `Kinematic` pontot True értékre állítani, majd a modellt elmenteni. A pályán látható színek egyenlőre nem egyeznek meg a Blender-ben létrehozottakkal, azonban ez kiküszöbölhető a `model.sdf` fájljának módosításával a következőképpen: a `visual` szekcióban a `material` részhez tartozó sorokat kitöröljük. Ezután már a Gazebo környezetében is pontosan látszanak a beállított színek.

### Kamera hozzáadása a robothoz
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
## 3. A feladat megoldásának menete
### A tanításhoz használandó képek elkészítése
A jól kondícionált tudás eléréséhez nagyméretű tanító adathatlmazra lesz szükség. Ha minden különböző vonalszínt és lehetséges trajektóriát szeretnénk reprezentálni változatos orientációkból, ahhoz a legkézenfekvőbb megoldás a pálya teljes bejárása közbeni fényképes dokumentálás. A robot kézi manőverezéséhez a távirányító node használható, mely a `W,A,S,D,X` billentyűk általi irányítást teszi lehetővé.
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
- Egy pillanatkép az említett dokumentálási módszer menetéről:
> ![Screenshot from 2024-04-11 18-56-40](https://github.com/4isSorin/Line_Following_kogrob/assets/167373493/2e4e80d7-1b2c-4756-a0ca-fad96196be46)

- Az elkészített felvételeket színtől és cselekvésti tervtől függően 10 mappába soroltuk, melyek a következőek:
> ![kép](https://github.com/4isSorin/Line_Following_kogrob/assets/167373493/9bcad1ca-1b42-4f9b-ad1f-9bed72927dfd)
- A mappák tartalma a következő képpen nézett ki:
> ![kép](https://github.com/4isSorin/Line_Following_kogrob/assets/167373493/d5064184-552a-4fb7-a6b0-76abbc00d37a)

### A neurális háló létrehozása
A hálózatunk egy LeNet-5 típusú konvulúciós neurális háló, melynek bemenetei 24x24 pixeles RGB-képek. A tanítás a `train_network.py` szkript segítségével törénik:
```python
# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Flatten, Dense, Conv2D, MaxPooling2D, LeakyReLU
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import __version__ as keras_version
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.random import set_seed
import tensorflow as tf
from sklearn.model_selection import train_test_split
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
    
#    model.add(Activation("relu"))  Ez volt eredetileg az aktivacios fuggveny
#    model.add(LeakyReLU(alpha=0.1))  Ez lett az uj
    
    model.add(Conv2D(20, (5, 5), padding="same", input_shape=inputShape))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second set of CONV => RELU => POOL layers
    model.add(Conv2D(50, (5, 5), padding="same"))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(500))
    model.add(LeakyReLU(alpha=0.1))

    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))	# softmax helyett sigmoid

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
    if label == 'fw_blue':
        label = 0
    elif label == 'fw_green':
        label = 3
    elif label == 'fw_yellow':
        label = 6
    elif label == 'right_blue':
        label = 2
    elif label == 'right_yellow':
        label = 8
    elif label == 'right_green':
        label = 5
    elif label == 'left_yellow':
        label = 7
    elif label == 'left_blue':
        label = 1
    elif label == 'left_green':
        label = 4
    else:
        label = 9
    labels.append(label)
    
    
# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
 
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=10)
testY = to_categorical(testY, num_classes=10)


# initialize the number of epochs to train for, initial learning rate,
# and batch size
EPOCHS  = 100	# Eredtileg 40
INIT_LR = 0.001
DECAY   = INIT_LR / EPOCHS
BS      = 80		# Batch size eredetileg 32

# initialize the model
print("[INFO] compiling model...")
model = build_LeNet(width=image_size, height=image_size, depth=3, classes=10)
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
- A tanítási folyamat eredményét az epochszám-pontosság grafikon ábrázolja.
> ![kép](https://github.com/4isSorin/Line_Following_kogrob/assets/167373493/58727cab-95ec-4698-9cee-618d5aedf0a9)
> > A tanítás annál jobb minél jobban közelíti a narancsárga görbét a piros illetve a kéket a zöld

- Miután a tanítás elvártnak megfelelő görbéket eredményez meg lehet kezdeni ennek gyakorlatban történő tesztelését. Ehhez az általunk testreszabott `line_follower_cnn.py`fájl külön terminálban történő futtatása szükséges:
```python
#!/usr/bin/env python3

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto
from tensorflow.keras import __version__ as keras_version
import tensorflow as tf

import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist
import rospy
import rospkg
try:
    from queue import Queue
except ImportError:
    from Queue import Queue
import threading
import numpy as np
import h5py
import time

# Set image size
image_size = 24

# Initialize Tensorflow session
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Initialize ROS node and get CNN model path
rospy.init_node('line_follower')

rospack = rospkg.RosPack()
path = rospack.get_path('turtlebot3_mogi')
model_path = path + "/network_model/model.best.h5"

print("[INFO] Version:")
print("OpenCV version: %s" % cv2.__version__)
print("Tensorflow version: %s" % tf.__version__)
keras_version = str(keras_version).encode('utf8')
print("Keras version: %s" % keras_version)
print("CNN model: %s" % model_path)
f = h5py.File(model_path, mode='r')
model_version = f.attrs.get('keras_version')
print("Model's Keras version: %s" % model_version)

if model_version != keras_version:
    print('You are using Keras version ', keras_version, ', but the model was built using ', model_version)

# Finally load model:
model = load_model(model_path)

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

        # Initialize published Twist message
        self.cmd_vel = Twist()
        self.cmd_vel.linear.x = 0
        self.cmd_vel.angular.z = 0
        self.last_time = time.time()

    def run(self):
        # Create a single OpenCV window
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("frame", 800,600)

        while True:
            self.image = self.queue.get()

            # Process the current image
            mask = self.processImage(self.image)

            # Add processed images as small images on top of main image
            result = self.addSmallPictures(self.image, [mask])
            cv2.imshow("frame", result)

            # Check for 'q' key to exit
            k = cv2.waitKey(1) & 0xFF
            if k in [27, ord('q')]:
                # Stop every motion
                self.cmd_vel.linear.x = 0
                self.cmd_vel.angular.z = 0
                pub.publish(self.cmd_vel)
                # Quit
                rospy.signal_shutdown('Quit')

    def processImage(self, img):

        image = cv2.resize(img, (image_size, image_size))
        image = img_to_array(image)
        image = np.array(image, dtype="float") / 255.0

        image = image.reshape(-1, image_size, image_size, 3)
        
        with tf.device('/gpu:0'):
            prediction = np.argmax(model(image, training=False))
                
        print("Prediction %d, elapsed time %.3f" % (prediction, time.time()-self.last_time))
        self.last_time = time.time()

        if prediction == 0: # Forward blue
            self.cmd_vel.angular.z = 0
            self.cmd_vel.linear.x = 0.2
        elif prediction == 1: # Left blue
            self.cmd_vel.angular.z = 0.2
            self.cmd_vel.linear.x = 0.05
        elif prediction == 2: # Right blue
            self.cmd_vel.angular.z = -0.2
            self.cmd_vel.linear.x = 0.05
        elif prediction == 3: # Forward green
            self.cmd_vel.angular.z = 0
            self.cmd_vel.linear.x = 0.3    
        elif prediction == 4: # Left green
            self.cmd_vel.angular.z = 0.2
            self.cmd_vel.linear.x = 0.05
        elif prediction == 5: # Right green
            self.cmd_vel.angular.z = -0.2
            self.cmd_vel.linear.x = 0.05
        elif prediction == 6: # Forward yellow
            self.cmd_vel.angular.z = 0
            self.cmd_vel.linear.x = 0.1
        elif prediction == 7: # Left yellow
            self.cmd_vel.angular.z = 0.2
            self.cmd_vel.linear.x = 0.05
        elif prediction == 8: # Right yellow
            self.cmd_vel.angular.z = -0.2
            self.cmd_vel.linear.x = 0.05        
        else: # Nothing
            self.cmd_vel.angular.z = -0.1
            self.cmd_vel.linear.x = 0.0

        # Publish cmd_vel
        pub.publish(self.cmd_vel)
        
        # Return processed frames
        return cv2.resize(img, (image_size, image_size))

    # Add small images to the top row of the main image
    def addSmallPictures(self, img, small_images, size=(160, 120)):
        x_base_offset = 40
        y_base_offset = 10

        x_offset = x_base_offset
        y_offset = y_base_offset

        for small in small_images:
            small = cv2.resize(small, size)
            if len(small.shape) == 2:
                small = np.dstack((small, small, small))

            img[y_offset: y_offset + size[1], x_offset: x_offset + size[0]] = small

            x_offset += size[0] + x_base_offset

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


queueSize = 1      
qMono = BufferQueue(queueSize)

bridge = CvBridge()

# Define your image topic
image_topic = "/camera/image/compressed"
# Set up your subscriber and define its callback
rospy.Subscriber(image_topic, CompressedImage, queueMonocular)

pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

# Start image processing thread
cvThreadHandle = cvThread(qMono)
cvThreadHandle.setDaemon(True)
cvThreadHandle.start()

# Spin until Ctrl+C
rospy.spin()
```

## 4. Eredmények
- A fájl az elvárásaink szerint teljesített: A robot felismerte a különböző színeket és ennek megfelelően különböző sebsségekkel haladt az adott pályarészeken. Itt fontos megjegyezni, hogy ugyanezt a rendszert ha más virtuális pályára, vagy valóságos pályára tanítanak be akkor az feltehetően képes hasonlóan precíz viselkedésre mint az általunk szimulált környezetben.
- A teljes rendszer működése az alábbi Youtube videón tekinthető meg:

<a href="https://www.youtube.com/watch?v=jogtECytDSQ">![kép](https://github.com/4isSorin/Line_Following_kogrob/assets/167373493/c59e7185-27f9-43be-9907-779edbcfed48)</a>
