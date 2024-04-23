# Line_Following_kogrob
Multiple colored line following with neural network using ROS and turtlebot
## Tartalomjegyzék
1. [Előkészületek](##1.-előkészületek)  
2. [Első lépések](##Első-lépések)  
3. [Feladat kidolgozása és megoldása](##3.-Feladat-kidolgozása-és-megoldása)
4. [Eredmányek](##4.-Eredmények)  
## 1. Előkészületek
- A projekt első lépéseként telepítettük az `Ubuntu 20.04`-verziószámú operációs rendszert `Oracle VM VirtualBox` szoftver segítségével virtuális környezetben.
> ![kép](https://github.com/4isSorin/Line_Following_kogrob/assets/167373493/5ffaaaae-4031-40e2-b34c-9afc5df87947)
> > Alap adatok a rendszerről

Ezután a friss felületre telepítettük a feladat megoldásához szükséges szoftvereket (mint például: `ROS Noetic` - `full desktop`csomag, `Python3.0`, `PyCharm Community Edition`). Majd elkezdtük letölteni a várhatóan szükséges Python könyvtárakat, melyek számát a feladat megoldása során szükség szerint tovább bővítettük. Az említett könyvtárak pontos nevei a dokumentációhoz csatolt kódok importálási szekciójában lesznek láthatóak.
## 2. Első lépések
### Projekt feladatának meghatározása
- Mielőtt tovább léptünk volna tisztáznunk kellett, hogy ponotosan milyen célt szeretnénk elérni a projekt során a vonalkövetés témakörön belül. A választás arra esett, hogy a robotnak képesnek kell lennie neurális hálót használva többszínű vonallal ellátott pályát követni úgy, hogy meg is különböztesse a tanított színeket és különböző színek esetén különböző sebességekkel haladjon.  
### Pálya elkészítése
- A vonalkövetéshez elengedhetetlen volt egy saját pálya amin betaníthatjuk és tesztelhetjük a továbbiakban a robotot. Ezt a pályát `Blender 4.0` segítségével készítettük el.
> ![kép](https://github.com/4isSorin/Line_Following_kogrob/assets/167373493/bc0be3e7-94a3-49f5-baa3-ff3c2224ecf9)
> > Kész pálya három különböző színnel
- A pályaát collada (.dae) formátumba exportáltuk ki, ugyanis ez kedvező lesz a továbbiakban a Gazebo-ba való importálás szempontjából.
- Gazeboba valo beillesztes
- nano .bashrc
- Elkészítettük a catkin workspace-ünket, ami az egyszerűség kedvéért a `bme_catkin_ws` nevet kapta.
## 3. Feladat megoldása
- Írjunk ilyen "érdekességek" részt a hibákról amik felemrültek? (mondjuk valszeg ez felesleges)
- Feladat megoldásának menetét leírni, csak konkrétan hogy milyen scripteket használtunk, melyik mit csinál, elmesélni a betanítást + pár kép és kész
## 4. Eredmények
[Youtube videó link](https://www.youtube.com/watch?v=jogtECytDSQ&t=114s)
