Laikapstākļu prognozēšanas lietotne

Pārskats
Šī lietotne ir laikapstākļu prognozēšanas rīks, kas paredz temperatūru, balstoties uz laikapstākļu parametriem, izmantojot mašīnmācīšanos. Tā iegūst laikapstākļu datus no API un izmanto apmācītu modeli, lai prognozētu temperatūru. Lietotne ir izstrādāta Python valodā un izmanto tādas bibliotēkas kā `tkinter` grafiskajam interfeisam un `scikit-learn` mašīnmācīšanās modelim.

Funkcijas
1. Laikapstākļu datu iegūšana: Reāllaika laikapstākļu datu iegūšana no Open-Meteo un vietas datu iegūšana no OpenStreetMap API.
2. Mašīnmācīšanās modelis: Gradientu uzlabojošais regresors, kas apmācīts ar sintētiskiem laikapstākļu datiem, lai prognozētu temperatūru.
3. Grafiskais interfeiss (GUI): Lietotāji var ievadīt pilsētu un datumu, lai saņemtu prognozi.
4. Mākoņu animācija: Dekoratīvs animēts mākoņu attēlojums interfeisā.
5. Kļūdu apstrāde: Informējoši kļūdu ziņojumi par nepareiziem ievadītajiem datiem vai API kļūmēm.

---

Instalācija un lietošana

Prasības
1. Python 3.8 vai jaunāka versija.
2. Nepieciešamās bibliotēkas:
   - `requests`
   - `pandas`
   - `datetime`
   - `tkinter`
   - `joblib`
   - `scikit-learn`

   Instalējiet bibliotēkas ar:
   ```bash
   pip install requests pandas scikit-learn
