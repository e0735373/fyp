
(define (problem maze) (:domain maze)
  (:objects
        loc-1-1 - location
	loc-1-2 - location
	loc-1-3 - location
	loc-1-4 - location
	loc-1-5 - location
	loc-1-6 - location
	loc-1-7 - location
	loc-1-8 - location
	loc-1-9 - location
	loc-2-1 - location
	loc-2-2 - location
	loc-2-3 - location
	loc-2-4 - location
	loc-2-5 - location
	loc-2-6 - location
	loc-2-7 - location
	loc-2-8 - location
	loc-2-9 - location
	loc-3-1 - location
	loc-3-2 - location
	loc-3-3 - location
	loc-3-4 - location
	loc-3-5 - location
	loc-3-6 - location
	loc-3-7 - location
	loc-3-8 - location
	loc-3-9 - location
	loc-4-1 - location
	loc-4-2 - location
	loc-4-3 - location
	loc-4-4 - location
	loc-4-5 - location
	loc-4-6 - location
	loc-4-7 - location
	loc-4-8 - location
	loc-4-9 - location
	loc-5-1 - location
	loc-5-2 - location
	loc-5-3 - location
	loc-5-4 - location
	loc-5-5 - location
	loc-5-6 - location
	loc-5-7 - location
	loc-5-8 - location
	loc-5-9 - location
	loc-6-1 - location
	loc-6-2 - location
	loc-6-3 - location
	loc-6-4 - location
	loc-6-5 - location
	loc-6-6 - location
	loc-6-7 - location
	loc-6-8 - location
	loc-6-9 - location
	loc-7-1 - location
	loc-7-2 - location
	loc-7-3 - location
	loc-7-4 - location
	loc-7-5 - location
	loc-7-6 - location
	loc-7-7 - location
	loc-7-8 - location
	loc-7-9 - location
	loc-8-1 - location
	loc-8-2 - location
	loc-8-3 - location
	loc-8-4 - location
	loc-8-5 - location
	loc-8-6 - location
	loc-8-7 - location
	loc-8-8 - location
	loc-8-9 - location
	loc-9-1 - location
	loc-9-2 - location
	loc-9-3 - location
	loc-9-4 - location
	loc-9-5 - location
	loc-9-6 - location
	loc-9-7 - location
	loc-9-8 - location
	loc-9-9 - location
	player-1 - player
  )
  (:init 
	(at player-1 loc-8-3)
	(clear loc-2-4)
	(clear loc-2-5)
	(clear loc-2-6)
	(clear loc-2-8)
	(clear loc-3-3)
	(clear loc-3-4)
	(clear loc-3-6)
	(clear loc-3-7)
	(clear loc-3-8)
	(clear loc-4-2)
	(clear loc-4-3)
	(clear loc-4-6)
	(clear loc-4-8)
	(clear loc-5-3)
	(clear loc-5-5)
	(clear loc-5-6)
	(clear loc-5-8)
	(clear loc-6-2)
	(clear loc-6-3)
	(clear loc-6-4)
	(clear loc-6-7)
	(clear loc-7-2)
	(clear loc-7-4)
	(clear loc-7-6)
	(clear loc-7-7)
	(clear loc-7-8)
	(clear loc-8-4)
	(clear loc-8-5)
	(clear loc-8-6)
	(clear loc-8-8)
	(is-goal loc-2-6)
	(move-dir-down loc-2-4 loc-3-4)
	(move-dir-down loc-2-6 loc-3-6)
	(move-dir-down loc-2-8 loc-3-8)
	(move-dir-down loc-3-3 loc-4-3)
	(move-dir-down loc-3-6 loc-4-6)
	(move-dir-down loc-3-8 loc-4-8)
	(move-dir-down loc-4-3 loc-5-3)
	(move-dir-down loc-4-6 loc-5-6)
	(move-dir-down loc-4-8 loc-5-8)
	(move-dir-down loc-5-3 loc-6-3)
	(move-dir-down loc-6-2 loc-7-2)
	(move-dir-down loc-6-4 loc-7-4)
	(move-dir-down loc-6-7 loc-7-7)
	(move-dir-down loc-7-4 loc-8-4)
	(move-dir-down loc-7-6 loc-8-6)
	(move-dir-down loc-7-8 loc-8-8)
	(move-dir-left loc-2-5 loc-2-4)
	(move-dir-left loc-2-6 loc-2-5)
	(move-dir-left loc-3-4 loc-3-3)
	(move-dir-left loc-3-7 loc-3-6)
	(move-dir-left loc-3-8 loc-3-7)
	(move-dir-left loc-4-3 loc-4-2)
	(move-dir-left loc-5-6 loc-5-5)
	(move-dir-left loc-6-3 loc-6-2)
	(move-dir-left loc-6-4 loc-6-3)
	(move-dir-left loc-7-7 loc-7-6)
	(move-dir-left loc-7-8 loc-7-7)
	(move-dir-left loc-8-4 loc-8-3)
	(move-dir-left loc-8-5 loc-8-4)
	(move-dir-left loc-8-6 loc-8-5)
	(move-dir-right loc-2-4 loc-2-5)
	(move-dir-right loc-2-5 loc-2-6)
	(move-dir-right loc-3-3 loc-3-4)
	(move-dir-right loc-3-6 loc-3-7)
	(move-dir-right loc-3-7 loc-3-8)
	(move-dir-right loc-4-2 loc-4-3)
	(move-dir-right loc-5-5 loc-5-6)
	(move-dir-right loc-6-2 loc-6-3)
	(move-dir-right loc-6-3 loc-6-4)
	(move-dir-right loc-7-6 loc-7-7)
	(move-dir-right loc-7-7 loc-7-8)
	(move-dir-right loc-8-3 loc-8-4)
	(move-dir-right loc-8-4 loc-8-5)
	(move-dir-right loc-8-5 loc-8-6)
	(move-dir-up loc-3-4 loc-2-4)
	(move-dir-up loc-3-6 loc-2-6)
	(move-dir-up loc-3-8 loc-2-8)
	(move-dir-up loc-4-3 loc-3-3)
	(move-dir-up loc-4-6 loc-3-6)
	(move-dir-up loc-4-8 loc-3-8)
	(move-dir-up loc-5-3 loc-4-3)
	(move-dir-up loc-5-6 loc-4-6)
	(move-dir-up loc-5-8 loc-4-8)
	(move-dir-up loc-6-3 loc-5-3)
	(move-dir-up loc-7-2 loc-6-2)
	(move-dir-up loc-7-4 loc-6-4)
	(move-dir-up loc-7-7 loc-6-7)
	(move-dir-up loc-8-4 loc-7-4)
	(move-dir-up loc-8-6 loc-7-6)
	(move-dir-up loc-8-8 loc-7-8)
	(oriented-down player-1)
  )
  (:goal (at player-1 loc-2-6))
)
