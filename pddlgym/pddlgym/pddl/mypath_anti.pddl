(define (domain path)
(:requirements :typing)
(:types NODE)
(:predicates
  (at ?l - NODE)
  (edge ?l1 - NODE ?l2 - NODE)
  (move ?l2 - NODE)
  (goal-at ?l - NODE)
)

; (:actions move)

(:action move
  :parameters (?l1 - NODE ?l2 - NODE)
  :precondition (and (at ?l1) (edge ?l1 ?l2) (move ?l2))
  :effect (and (at ?l2) (not (at ?l1)))
)
)
