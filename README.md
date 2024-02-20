Dataset: FMA

Se va folosi folderul 'Samples' unde se vor regăsi cele 8000 de fișiere încă neetichetate. În proiect trebuie să fie prezent și fișierul tracks.csv. Se rulează funcția de etichetare label_function.py, apoi funcția de preprocesare preprocessor.py. Dimensiunile imaginilor pot fi setate în funcție de preferințele utilizatorului, la fel și numărul de partiții (pentru cazul de față, am folosit 10 partiții per spectrogramă, deci 10*8*1000 = 80000 de imagini.

Se va implementa modelul CNN folosind Keras. Pentru partea de preprocesare s-a folosit Librosa.
