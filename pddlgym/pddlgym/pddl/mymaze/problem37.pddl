
(define (problem mymaze) (:domain maze)
  (:objects
        loc-1-1 - location
	loc-1-2 - location
	loc-1-3 - location
	loc-1-4 - location
	loc-1-5 - location
	loc-1-6 - location
	loc-1-7 - location
	loc-2-1 - location
	loc-2-2 - location
	loc-2-3 - location
	loc-2-4 - location
	loc-2-5 - location
	loc-2-6 - location
	loc-2-7 - location
	loc-3-1 - location
	loc-3-2 - location
	loc-3-3 - location
	loc-3-4 - location
	loc-3-5 - location
	loc-3-6 - location
	loc-3-7 - location
	loc-4-1 - location
	loc-4-2 - location
	loc-4-3 - location
	loc-4-4 - location
	loc-4-5 - location
	loc-4-6 - location
	loc-4-7 - location
	loc-5-1 - location
	loc-5-2 - location
	loc-5-3 - location
	loc-5-4 - location
	loc-5-5 - location
	loc-5-6 - location
	loc-5-7 - location
	loc-6-1 - location
	loc-6-2 - location
	loc-6-3 - location
	loc-6-4 - location
	loc-6-5 - location
	loc-6-6 - location
	loc-6-7 - location
	loc-7-1 - location
	loc-7-2 - location
	loc-7-3 - location
	loc-7-4 - location
	loc-7-5 - location
	loc-7-6 - location
	loc-7-7 - location
	player-1 - player
  )
  (:init 
	(at player-1 loc-3-5)
	(clear loc-2-2)
	(clear loc-2-3)
	(clear loc-2-5)
	(clear loc-3-2)
	(clear loc-3-4)
	(clear loc-3-6)
	(clear loc-4-2)
	(clear loc-4-3)
	(clear loc-4-4)
	(clear loc-4-6)
	(clear loc-5-2)
	(clear loc-5-4)
	(clear loc-5-6)
	(clear loc-6-2)
	(clear loc-6-3)
	(clear loc-6-5)
	(clear loc-6-6)
	(is-goal loc-6-5)
	(move-dir-down loc-2-2 loc-3-2)
	(move-dir-down loc-2-5 loc-3-5)
	(move-dir-down loc-3-2 loc-4-2)
	(move-dir-down loc-3-4 loc-4-4)
	(move-dir-down loc-3-6 loc-4-6)
	(move-dir-down loc-4-2 loc-5-2)
	(move-dir-down loc-4-4 loc-5-4)
	(move-dir-down loc-4-6 loc-5-6)
	(move-dir-down loc-5-2 loc-6-2)
	(move-dir-down loc-5-6 loc-6-6)
	(move-dir-left loc-2-3 loc-2-2)
	(move-dir-left loc-3-5 loc-3-4)
	(move-dir-left loc-3-6 loc-3-5)
	(move-dir-left loc-4-3 loc-4-2)
	(move-dir-left loc-4-4 loc-4-3)
	(move-dir-left loc-6-3 loc-6-2)
	(move-dir-left loc-6-6 loc-6-5)
	(move-dir-right loc-2-2 loc-2-3)
	(move-dir-right loc-3-4 loc-3-5)
	(move-dir-right loc-3-5 loc-3-6)
	(move-dir-right loc-4-2 loc-4-3)
	(move-dir-right loc-4-3 loc-4-4)
	(move-dir-right loc-6-2 loc-6-3)
	(move-dir-right loc-6-5 loc-6-6)
	(move-dir-up loc-3-2 loc-2-2)
	(move-dir-up loc-3-5 loc-2-5)
	(move-dir-up loc-4-2 loc-3-2)
	(move-dir-up loc-4-4 loc-3-4)
	(move-dir-up loc-4-6 loc-3-6)
	(move-dir-up loc-5-2 loc-4-2)
	(move-dir-up loc-5-4 loc-4-4)
	(move-dir-up loc-5-6 loc-4-6)
	(move-dir-up loc-6-2 loc-5-2)
	(move-dir-up loc-6-6 loc-5-6)
	(oriented-down player-1)
  )
  (:goal (at player-1 loc-6-5))
)
