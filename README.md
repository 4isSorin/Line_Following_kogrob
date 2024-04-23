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
- A `gazebo` parancs segítségével megnyílik a környezet, majd az Edit menüpont alatt kiválasztjuk a Model Editor-t, melyen belül a Custom Shapes Add gombjára kattintva betallózzunk a létrehozott fájlt. A Link Inspector menüt előhozva célszerű a Kinematic pontot True értékre állítani, majd a modellt elmenteni. A pályán látható színek egyenlőre nem egyeznek meg a Blender-ben létrehozottakkal, azonban ez kiküszöbölhető a model.sdf fájljának módosításával a következőképpen: a <visual> szekcióban a <material> részhez tartozó sorokat kitöröljük. Ezután már a Gazebo környezetében is pontosan látszanak a beállított színek.
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
[tanítási képek beszúrása]
- nano .bashrc
- Elkészítettük a catkin workspace-ünket, ami az egyszerűség kedvéért a `bme_catkin_ws` nevet kapta.
## 3. Feladat megoldása
- Írjunk ilyen "érdekességek" részt a hibákról amik felemrültek? (mondjuk valszeg ez felesleges)
- Feladat megoldásának menetét leírni, csak konkrétan hogy milyen scripteket használtunk, melyik mit csinál, elmesélni a betanítást + pár kép és kész
## 4. Eredmények
[Youtube videó link](https://www.youtube.com/watch?v=jogtECytDSQ&t=0s)
