(define (domain sudoku)
(:requirements :typing)
(:types LOCATION NUMBER)
(:predicates
  (at ?l - LOCATION)
  (value ?l - LOCATION ?n - NUMBER)
  (next-location ?l1 - LOCATION ?l2 - LOCATION)
  (move ?n - NUMBER)
  (official-solution ?l - LOCATION ?n - NUMBER)
)

; (:actions move)

(:action move
  :parameters (?l1 - LOCATION ?l2 - LOCATION ?n - NUMBER)
  :precondition (and (move ?n) (at ?l1) (next-location ?l1 ?l2) )
  :effect (and (at ?l2) (not (at ?l1)) (value ?l1 ?n))
)
)
