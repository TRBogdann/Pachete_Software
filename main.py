import streamlit as st
import nbformat
import pandas as pd

def printNoteBook(path,title,small_title=False):
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    
    if small_title:
        st.write(title)
    else:
        st.title(title)
        
    st.components.v1.html(html,scrolling=True,height=1000,width=1000)

st.set_page_config(page_title="Proiect Pachete Software",page_icon=":raccon:")

def printCode(path,title,small_title=False,language="python"):
    if small_title:
        st.write(title)
    else:
        st.title(title)
    
    with open(path,"r") as file:
        python_code = file.read()
        
    st.code(python_code, language=language)
st.title("GitHub")
st.write("Link Cod Complet: https://github.com/TRBogdann/Licenta")
st.title("Seturi de date")
st.write("Brain Tumor Classification: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset")
st.write("Brats: https://www.kaggle.com/datasets/awsaf49/brats2020-training-data")
st.write("Chest X-ray: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database")
st.write("Retinal OCT: https://www.kaggle.com/datasets/obulisainaren/retinal-oct-c8")
st.write("Leukemia: https://www.kaggle.com/datasets/mehradaria/leukemia")
st.write("CKD: https://www.kaggle.com/datasets/mansoordaku/ckdisease")
st.write("Diabetes: https://www.kaggle.com/datasets/mathchi/diabetes-data-set")
st.write("Anemia: https://www.kaggle.com/datasets/biswaranjanrao/anemia-dataset")
st.title("Introducere")
st.write("Pentru realizarea acestui proiect au fost publicate date de la institutii medicale ce au fost mai apoi republicate pe alte site-uri(cum ar fi kaggle). Datele au fost utilizate pentru a crea model ce pot fi utilizate de clinici pentru a asista medicul in luarea decizilor si pentru a grabi procesul de diagnosticare.De asemenea modelele pot prelucra datele pentru a fi mai usor de interpretat de catre medic.")

st.title("Notebooks")

printNoteBook("./python/brain_tumor_classification.html","Brain Tumor Classification")
st.write("Permite detectarea urmatoarelor tipuri de tumori:")
st.write("-Glioma: Sunt tumori ce se formeaza din celule gliale, ce protejeaza si sustin neuronii. Sunt tumori agresive ce pot invada usor celelalte parti ale creierului si duc la distrugerea neuronilor")
st.write("-Tumora Pineala: Apare la glanda pineala si cauzeaza problema in reglarea hormonilor si a ritmului inimii")
st.write("-Meningioma: Este o tumora ce apare la nivelul meningelui(tesutul ce acopera creierul si maduva spinarii). Sunt tumori ce se extind lent si adesea nu cauzeaza complicatii")
st.write("Pentru rezolvarea problemei s-a utilizat o retea neuronala convolutionala cu 4 straturi. Primul strat contine o singura convolutie, urmatoarele sunt formate din 2 convolutii")
st.write("Modelul are o acuratete de 98.4%")
printCode("./python/fake_segmentation.py","Pseudo segmentare folosing activarile convolutilor")
st.write("Feature Map")
st.write("Odata ce modelul a fost antrenat putem extrage output-ul straturilor intermediare pentru a intelege mai bine ce parti din imagine sunt utilizate de catre model pentru luarea decizilor. Tot odata putem folosi aceste output uri pentru a segmenta imaginile, fiind scoase in evidenta zonele de inteles")
st.image("feature_map.png",width=1000)

st.title("Brain Tumor Segmentation")
printNoteBook("./python/Brats2020_Best_Model.html","Antrenare",True)
printNoteBook("./python/test-brats.html","Testare",True)
st.write("Modelul a fost antrenat folosind datele de la Multimodal Brain Tumor Segmentation. Modelul poate detecta anomalii ce apar la nivelul creierului in urma examniarii tomografelor. Poate fi folosit pentru a monitoriza evolutia starii de sanatate a unui pacient si pentru a scoate in evidenta regiunile de interes pentru medic. Sunt utilizate 3 masti de segmentare:")
st.write("-Necroza Tumorala sau Centrul tumorei: Parte a tumorei formate din celule ce au murit ca urmare a extinderii tumorei")
st.write("-Edema: Este o umflatura cauzata de acumularea de lichid in tesuturi. Poate aparea in urma unei leziuni, atac cerebral sau datorita formarii unei tumori")
st.write("-Enhancing: Zona in care creierul este inflat sau in care vascularizarea este normala. Nu poate fi vazuta in mod normala si este necesara administrarea unui ser ce mareste contrastul acestei zona")
st.write("Arhitectura modelului este unde testul de complicata. Este folosit un model de tip UNET la care s-a mai adaugat o dimensiune pentru a putea reprezenta legaturile dintre slice-urile tomografiei. In decodificare au fost adaugate module de atentie pentru a-i permite modelului sa separe creierul de resutul imaginii si mai apoi sa separe zona de interes si sa o segmenteze")
printCode("./python/overlay.py","Overlay pentru detectie")
st.write("Overlay")
st.write("Zone evidentiate:")
st.image("overlay.png")

st.write("Urmatoarele doua modele folosesc arhitecturi similare arhitectura primului model. Difera doar adancimea retelei")
printNoteBook("./python/chest.html","Chest X-Ray")
st.write("Folosit pentru a detecta cazurile de COVID si Pneumonie cauzata de aceasta. Poate fi folosit si pentru a semnala si prezenta altor probleme pulmonare, dar nu poate sa spuna cu certitudine care sunt acestea (Sunt clasificate ca 'Lung Opacity')")
st.write("Acuratetia initiala a fost de 88% si a crecut la 90% dupa adaugarea unui nou strat. Dupa ce am implementat corectia de culoare pe baza histogramei acuratea a crescut la 94%. Am putea creste si mai mult acuratetea prin utilizarea unui model cu mai multe straturi (RESNET de exemplu).Modelul prezinta problema la diferentiarea plamanilor ce prezinta o opacitate usoare de cei sanatosi")
printNoteBook("./python/eye.html","Eye Disease clasification")
st.write("Foloseste tomografii optice pentru a detecta probleme oculare")
st.write("Modelul are o acuratete de 98%, avand problema in detectarea retinopatiei")
st.title("Leucemia")
st.write("Programul utilizeaza doua modele , unul ce se ocupa de segmentarea imaginii si unul ce se ocupa de segmentarea acesteia")
printNoteBook("./python/leucocite_seg.html","Leucemia I (Segmentation)")
st.write("Se foloseste arhitectura UNET pentru segmentare.Diagnosticul leucemiei se face pe baza numarului de celule albe,tipul celulelor si formei acestora, de aceea programul scoate in evidenta globulele albe pentru a face imaginea mai usor de interpretat.")
st.title("Leucemia II (Classification)")
printCode("./python/cod_antrenare.py","Cod Antrenare")
printCode("./python/test_leucemie_final.py","Cod Testare")

st.write("Output")
df_cm = pd.read_csv('./confusion_matrix.csv',index_col=0)
st.dataframe(df_cm)

correct_predictions = df_cm.values.diagonal().sum()
total_predictions = df_cm.values.sum()

accuracy = correct_predictions / total_predictions
st.write(f"Accuracy: {accuracy:.2%}")

#CDK
st.title("Tabele")
df_ckd = pd.read_csv("./python/kidney_disease.csv",index_col=0)
df_cf_ckd = pd.read_csv("./conf_ckd.csv",index_col=0)
df_res_ckd = pd.read_csv("./result.csv",index_col=0)

st.title("CDK - Insuficienta Renala")
st.write("Tabel Initial")
st.write("Tabelul initial contine multe valori lipsa. Dorim o completarea valorilor lipsa si gasirea celui mai bun model")
st.dataframe(df_ckd)
printCode("./python/ckd_fill.py","Completare Valori Lipsta",True)
st.write("Matrice de corelatie")
st.write("Matricea de corelatie arata corelatii intre coloane. Putem determina valorile lipsa pe baza realatiolor dintre acestea")
st.dataframe(df_cf_ckd)
printCode(" a./python/ckd.py","Gasirea celui mai bun model",True)
st.dataframe(df_res_ckd)

st.title("SAS")

st.title("N. Inst. of Diabetes & Diges. & Kidney Dis Dataset")
printCode('./sas/prog1/import.txt',"Importare Date",True,"sas")
printCode('./sas/prog1/corrstats.txt',"Statistici si Corelatii",True,"sas")
st.image('./sas/prog1/stats.png')
st.image('./sas/prog1/corelatii.png')
printCode('./sas/prog1/distributii.txt',"Distributii variabile categorice",True,"sas")
row_dist = st.columns(3)
row_dist[0].container().image("./sas/prog1/dist1.png")
row_dist[1].container().image("./sas/prog1/dist2.png")
row_dist[2].container().image("./sas/prog1/dist3.png")
printCode('./sas/prog1/pregatire.txt','Data Cleaning',True,"sas")
printCode('./sas/prog1/regresie.txt','Regresie',True,"sas")
row_dist = st.columns(2)
row_dist[0].container().image("./sas/prog1/regresie1.png")
row_dist[1].container().image("./sas/prog1/regresie2.png")
st.image("./sas/prog1/regresie3.png")
printCode('./sas/prog1/rezultat.txt','Rezultat',True,"sas")
st.image("./sas/prog1/result.png",width=1000)

st.title("Anemie")
st.write("Similar cu exemplul anterior se doreste determinarea celui mai bun model")
printCode('./sas/prog2/import.txt',"Importare Dare",True,"sas")
printCode('./sas/prog2/corrstats.txt','Statistici si Corelatii',True,"sas")
st.image('./sas/prog2/stats.png',width=1000)
st.image('./sas/prog2/corr.png',width=1000)
printCode('./sas/prog2/standardizare.txt','Standardizare',True,"sas")
printCode('./sas/prog2/impartire.txt',"Impartirea in subseturi de antrenament si test",True,"sas")
printCode('./sas/prog2/lda.txt',"LDA",True,"sas")
st.image('./sas/prog2/lda.png',width=1000)
st.image('./sas/prog2/lda2.png',width=1000)
st.write("In urma analizei output-ului putem oberva ca Homeglobina separa cel mai bine setul de date, avand cea mai mare pondere")
printCode('./sas/prog2/arbore_simplu.txt',"Arbore Simplu",True,"sas")
row_dist = st.columns(2)
st.image("./sas/prog2/arbore_o1.png")
row_dist[0].container().image("./sas/prog2/arbore_o2.png")
row_dist[1].container().image("./sas/prog2/arbore_o3.png")
st.write("Interpretare arbore")
st.write("Nivel 1: Atunci cand cantitatea de homoglobina este mai mare decat cea medie pacientul nu sufera de anemie")
st.write("Nivel 2: Daca nivelul homoglobinei este mai mic cu 0.7 decat cel mediu atunci pacientul sufera de anemie")
st.write("Nivel 3: Daca nivelul hemoglobinei este mai mic cu <0.7: ")
st.write("Femeie: Nu are anemie")
st.write("Barbat: are sanse de 95% de anemie (Existenta subramurilor)")
printCode('./sas/prog2/arbore_lda.txt',"Arbore LDA",True,"sas")
row_dist = st.columns(2)
st.image("./sas/prog2/arbore_lda1.png")
row_dist[0].container().image("./sas/prog2/arbore_lda2.png")
row_dist[1].container().image("./sas/prog2/arbore_lda3.png")
printCode('./sas/prog2/metrice.txt',"Matrice de confuzie",True,"sas")
st.image("./sas/prog2/conf.png")
printCode('./sas/prog2/rezultat.txt',"Rezultat",True,"sas")
st.image("./sas/prog2/rezultat.png",width=1000)
st.write("Se alege arborele simplu")
