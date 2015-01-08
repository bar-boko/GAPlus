g1_member(X):0.6*a*b<-g1_member(Y):b&p(Y):1&friend(Y,X):a&p(X):1
g1_member(X):0.3*a*b<-g1_member(Y):b&p(Y):1&friend(Y,X):a&q(X):1
g2_member(X):0.3*a*b<-g2_member(Y):b&q(Y):1&friend(Y,X):a&p(X):1
g2_member(X):0.4*a*b<-g2_member(Y):b&q(Y):1&friend(Y,X):a&q(X):1
g1_member(X):0.2*b<-g2_member(X):b&p(X):1
g2_member(X):0.4*b<-g1_member(X):b&q(X):1
