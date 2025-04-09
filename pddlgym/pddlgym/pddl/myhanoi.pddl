(define (domain myhanoi)
  (:requirements :strips)
  (:types PEG OBJ)
  (:predicates
  (clear ?x - OBJ)
  (on ?x - OBJ ?y - OBJ)
  (smaller ?x - OBJ ?y - OBJ)
  (obj-at-peg ?obj - OBJ ?peg - PEG)
  (move ?frompeg - PEG ?topeg - PEG)
  )

  ; (:actions move)

  (:action move
    :parameters (?frompeg - PEG ?topeg - PEG ?disc - OBJ ?from - OBJ ?to - OBJ)
    :precondition (and (move ?frompeg ?topeg) (smaller ?to ?disc) (on ?disc ?from) (clear ?disc) (clear ?to) (obj-at-peg ?disc ?frompeg) (obj-at-peg ?from ?frompeg) (obj-at-peg ?to ?topeg))
    :effect  (and (clear ?from) (on ?disc ?to) (not (on ?disc ?from)) (not (clear ?to))
    (not (obj-at-peg ?disc ?frompeg)) (obj-at-peg ?disc ?topeg) (obj-at-peg ?from ?frompeg) (obj-at-peg ?to ?topeg))
    )
  )
