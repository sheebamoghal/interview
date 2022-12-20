# Candidate Interview Prediction
The main objective of this is to predict the status of a candidate's interview so that recruiters can check the sanity of the interview & find if the interview was biased.

# Features 

-Independant values: 
    
1  Interview Id : Id for the interview
2  Candidate Id : Id for the candidate
3  Interviewer Id : Id for the interviewer
4  Profile : Profile type
5  S.L.R.C (Speak to Listen Ratio Candidate): It is the ratio of speaking time to the listening time of the Candidate.
6  S.L.R.I (Speak to Listen Ratio Interviewer): It is the ratio of speaking time to the listening time of the Interviewer.
7  A.T.T Avg Turn Time (Interactivity Time): It is the average amount of time in which single interaction happens between the Interviewer and the Candidate.
8  L.M.I (Longest Monologue Interviewer): It is the longest amount of time that the interviewer spoke continuously.
9  L.M.C (Longest Monologue Candidate): It is the longest amount of time that the candidate spoke continuously.
10 S.R (SILENCE RATIO):It is the percentage of time when no one had spoken
11 L.J.T.C (Late Joining Time Candidate): It is the amount of time a candidate joined late for the interview call.
12 L.J.T.I (Late Joining Time Interviewer): It is the amount of time the interviewer joined late for the interview call.
13 N.I.C(Noise Index Candidate): Percentage of Background Noise present on the candidate side.
14 N.I.I(Noise Index Interviewer): Percentage of Background Noise present on the interviewer’s side.
15 S.P.I - Speaking Pace interviewer: Average Number of words spoken per minute.
16 S.P.C - Speaking Pace Candidate: Average Number of words spoken per minute.
17 L.A.I - Live Absence interviewer: It is the percentage of time the interviewer was not present in the video call.
18 L.A.C - Live Absence candidate: It is the percentage of time the candidate was not present in the video call.
19 Q.A - Question asked during the interview
20 P.E.I - Perceived Emotion Interviewer: It is the perceived emotion of Interviewer which can be either Positive or Negative
21 P.E.C - Perceived Emotion Candidate: It is the perceived emotion of candidate which can be either Positive or Negative
22 Compliance ratio - Does the interviewer follow the structure that has been guided by the company. It is the ratio of questions assigned to the number of questions which were actually asked.
23 Interview Duration : For how much time interview happened
24 Interview Intro :Does interviewer give the self-introduction to the candidate or not
25 Candidate Intro :Does candidate give the self-introduction to the interviewer or not
26 Opp to ask : Does the interviewer give a chance to the candidate to ask questions at the end.

-Dependant values: 
#15 Status : Status of the candidate.

# Libraries Used
Numpy

Pandas

Seaborn

Matplotlib

Sklearn

Time

# Lessons Learned
This was an interesting topic to work as we used a lot of bagging and boosting techniques in addition to tradition regression techniques.

# Extra Information
This problem statement was shared with us during the first stage of Imarticus Data Science Hackathon, wherein I received the 3rd position amongst 40+ students in our professional certification course in association with KPMG.
