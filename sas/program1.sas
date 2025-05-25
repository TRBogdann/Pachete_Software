PROC IMPORT DATAFILE="/home/u64222239/proiect/dataIN/diabetes_dataset.csv"
    OUT=diabetes
    DBMS=csv
    REPLACE;
    GUESSINGROWS=MAX;
RUN;

* Corelatii; 
DATA cleaned_table;
    SET diabetes;
    IF cmiss(of _all_) THEN delete;
RUN;
PROC CORR DATA=cleaned_table PLOTS=matrix(HISTOGRAM);
    VAR _numeric_;
RUN;

* Distributia variabilelor categorice;
%MACRO COUNT_PLOT(var);
PROC SGPLOT DATA=cleaned_table;
    VBAR &var;
  	TITLE "Distributie &VAR";
run;
%MEND;

%COUNT_PLOT(FamilyHistory);
%COUNT_PLOT(Hypertension);
%COUNT_PLOT(DietType);

*Stergem coloanele care nu sunt relevante; 
DATA diabetes_selected;
    SET cleaned_table;
    DROP HbA1c WaistCircumference HipCircumference MedicationUse DietType Hypertension;
RUN;

*Stergere outliere;
%MACRO REMOVE_OUTLIERS(var);
    PROC UNIVARIATE DATA=diabetes_selected NOPRINT;
        VAR &var;
        OUTPUT OUT=stats_&var PCTLPTS=25 75 PCTLPRE=P_;
    RUN;

    DATA with_flags;
        IF _n_ = 1 THEN SET stats_&var;
        SET diabetes_selected;
        IQR = P_75 - P_25;
        Lower = P_25 - 1.5 * IQR;
        Upper = P_75 + 1.5 * IQR;
        is_outlier = (&var < Lower OR &var > Upper);
    RUN;

    PROC SQL NOPRINT;
        SELECT COUNT(*) INTO :total_obs FROM with_flags;
        SELECT COUNT(*) INTO :outlier_count FROM with_flags WHERE is_outlier = 1;
    QUIT;

    %IF %SYSEVALF((&outlier_count / &total_obs) < 0.01) %THEN %DO;
        DATA diabetes_selected;
            SET with_flags;
            IF is_outlier = 0;
            DROP IQR Lower Upper is_outlier P_25 P_75;
        RUN;
    %END;
    %ELSE %DO;
        DATA diabetes_selected;
            SET with_flags;
            DROP IQR Lower Upper is_outlier P_25 P_75;
        RUN;
    %END;

%MEND;

%REMOVE_OUTLIERS(Glucose);
%REMOVE_OUTLIERS(BMI);
%REMOVE_OUTLIERS(Age);
%REMOVE_OUTLIERS(Pregnancies);
%REMOVE_OUTLIERS(BloodPressure);
%REMOVE_OUTLIERS(HDL);
%REMOVE_OUTLIERS(WHR);
%REMOVE_OUTLIERS(Triglycerides);

* Standardizare;
%MACRO STANDARDIZE(DATA=, OUT=, TARGET=);
    PROC CONTENTS DATA=&DATA OUT=_VARS(KEEP=NAME TYPE) NOPRINT; RUN;

    PROC SQL NOPRINT;
        SELECT NAME INTO :NUM_VARS SEPARATED BY ' '
        FROM _VARS
        WHERE TYPE = 1 AND UPCASE(NAME) NE "%UPCASE(&TARGET)";
    QUIT;

    PROC STANDARD DATA=&DATA MEAN=0 STD=1 OUT=&OUT;
        VAR &NUM_VARS;
    RUN;

    DATA &OUT;
        MERGE &OUT(KEEP=&NUM_VARS) &DATA(KEEP=&TARGET);
    RUN;
%MEND;

%STANDARDIZE(DATA=diabetes_selected, OUT=standardized, TARGET=Outcome);

*Impartire Test-Antrenament;
PROC SURVEYSELECT DATA=standardized OUT=training_test SEED=42
    SAMPRATE=0.75 OUTALL;
run;

DATA train test;
    SET training_test;
    if selected then output train;
    else output test;
RUN;

* Regresie;
PROC LOGISTIC DATA=train;
    MODEL Outcome(EVENT='1') = _numeric_ / SELECTION=stepwise;
    SCORE DATA=test OUT=reg_out;
    TITLE "Rezultat Regresie";
RUN;

* Rezultat;
PROC FREQ data=reg_out;
    tables Outcome*P_1 / nopercent norow nocol;
    title "Matrice de confuzie";
RUN;

