# MAMO3200 Eksamen
I dette prosjektet har jeg modellert og presentert mine numeriske løsninger av dobbeltsplate eksperimentet

Det jeg har skrevet klart her er dobbelsplate eksperimentet med initialtilstand gaussisk bølge

Det er en dobbel spalte modellert av ett høyt potensial

Randbetingelsene er gitt ved dirichlet med null på randen

Jeg tester 4 forskjellige løsningsmetoder hvor 3 av dem er eksplisitte (forward_finite_difference, runge kutta 4 og leapfrog) og vurderer stabiliteten deres
Jeg tester også en stabil metode Crank Nichelson.

Jeg har laget tilhørende presentasjon fra en notebook som jeg brukte på eksamen

Selve programmeringen er delt i tre. Simulering, visualisering og en main. Mainen bringer de to andre delene sammen.