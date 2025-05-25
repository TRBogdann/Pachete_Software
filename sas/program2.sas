
proc import datafile="/home/u64222239/proiect/dataIN/anemia.csv"
    out=anemia dbms=csv replace;
    guessingrows=max;
run;

* Statistici set de date;
proc contents data=anemia; run;
proc means data=anemia; run;

* Corelatii;
proc corr data=anemia plots=matrix(histogram);
    var _numeric_;
run;

* Standardizare;
proc standard data=anemia mean=0 std=1 out=anemia_std;
    var Hemoglobin MCH MCHC MCV;
run;

* Impartirea setului in subset de antrenare si test;
proc surveyselect data=anemia_std out=split seed=42
    samprate=0.75 outall method=srs;
run;

data train test;
    set split;
    if selected then output train;
    else output test;
    drop selected;
run;

* LDA;
proc discrim data=train outstat=lda_stats outd=train_lda pool=yes method=normal;
    class Result;
    var Hemoglobin MCH MCHC MCV;
run;

proc discrim data=test testdata=test testout=test_lda
    method=normal pool=yes;
    class Result;
    var Hemoglobin MCH MCHC MCV;
run;

data test_lda_tree;
	set test_lda;
	keep Gender '0'n '1'n Result;
run;

data train_lda_tree;
	set train_lda;
	keep Gender '0'n '1'n Result;
run;

* Arbore Initial;
proc hpsplit data=train;
	class Result Gender;
    model Result = Gender Hemoglobin MCH MCHC MCV;
    prune costcomplexity;
    code file='/home/u64222239/proiect/dataOUT/tree_orig_score.sas';
run;

*Arbore LDA;
data scored_tree_orig;
    set test;
    %include '/home/u64222239/proiect/dataOUT/tree_orig_score.sas';
run;

proc hpsplit data=train_lda;
    class Gender Result;
    model Result = Gender '0'n '1'n;
    prune costcomplexity;
    code file='/home/u64222239/proiect/dataOUT/tree_lda_score.sas';
run;

*Matrice de confuzie LDA;
proc freq data=test_lda;
    tables Result*_INTO_ / nopercent norow nocol;
    title "Matrice de confuzie LDA";
run;

*Scor Test;
data scored_tree_lda;
    set test_lda;
    %include '/home/u64222239/proiect/dataOUT/tree_lda_score.sas';
run;

*Matrice de confuzie Arbore Simplu;
proc freq data=scored_tree_orig;
    tables Result*P_Result1 / norow nocol nopercent;
    title "Matrice de confuzie TREE";
run;

*Matrice de confuzie Arbore LDA;
proc freq data=scored_tree_lda;
    tables Result*P_Result1 / norow nocol nopercent;
	title "Matrice de confuzie TREE_LDA";
run;

*Acuratete modele;
proc sql noprint;
    select mean(Result = P_Result1) 
    into :acc_tree_orig
    from scored_tree_orig;
quit;

proc sql noprint;
    select mean(Result = _INTO_) 
    into :acc_lda
    from test_lda;
quit;

proc sql noprint;
    select mean(Result = P_Result1) 
    into :acc_tree_lda
    from scored_tree_lda;
quit;

data accuracy_array;
    array acc[3] _temporary_;
    acc[1] = &acc_tree_orig;
    acc[2] = &acc_lda;
    acc[3] = &acc_tree_lda;

    do i = 1 to 3;
        Accuracy = acc[i];
        Model = choosec(i, 'Arbore Simplu', 'LDA', 'Arbore+LDA');
        output;
    end;

    keep Model Accuracy;
run;

proc print data=accuracy_array label;
    title "Acuratetea Modelor Comparate";
    format Accuracy percent8.2;
run;