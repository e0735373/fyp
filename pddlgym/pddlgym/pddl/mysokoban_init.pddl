(define (domain typed-sokoban)
(:requirements :typing)
(:types LOC DIR BOX)
(:predicates
             (at-robot ?l - LOC)
             (at ?o - BOX ?l - LOC)
             (adjacent ?l1 - LOC ?l2 - LOC ?d - DIR)
             (clear ?l - LOC)
             (move ?v0 - DIR)
             (is-goal ?l - LOC)
)

; (:actions move)

(:action move
:parameters (?from - LOC ?to - LOC ?dir - DIR)
:precondition (and (move ?dir) (clear ?to) (at-robot ?from) (adjacent ?from ?to ?dir))
:effect (and (at-robot ?to) (not (at-robot ?from)))
)


(:action push
:parameters  (?rloc - LOC ?bloc - LOC ?floc - LOC ?dir - DIR ?b - BOX)
:precondition (and (move ?dir) (at-robot ?rloc) (at ?b ?bloc) (clear ?floc)
	           (adjacent ?rloc ?bloc ?dir) (adjacent ?bloc ?floc ?dir))

:effect (and (at-robot ?bloc) (at ?b ?floc) (clear ?bloc)
             (not (at-robot ?rloc)) (not (at ?b ?bloc)) (not (clear ?floc)))
)
)


