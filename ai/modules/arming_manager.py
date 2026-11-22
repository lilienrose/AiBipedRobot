import time
import cv2
import numpy as np

#Třída, která slouží jako bezpečnostní pojistka
#která zajistí, že robot nereaguje na náhodné pohyby
#tím, že gesto PEACE po 1,5 s ho odemkne či zamkne
#a po 5s se zamkne po době nečinnosti automaticky
class ArmingManager:
    def __init__(
        self,
        arm_gesture: str = "PEACE",
        hold_seconds: float = 1.5,
        auto_disarm_timeout: float = 5.0,
    ):
        self.arm_gesture = arm_gesture
        self.hold_seconds = hold_seconds
        self.auto_disarm_timeout = auto_disarm_timeout

        self.is_armed = False
        self.arm_start_time: float | None = None
        self.last_hand_seen_time: float = time.time()
        self.toggle_cooldown_until: float = 0.0

#  Okamžite uzamčení bez odpočítávání
    def force_disarm(self) -> None:
        """Okamžité bezpečnostní uzamčení (např. při dvou rukách v záběru)."""
        self.is_armed = False
        self.arm_start_time = None
# Volá se u každého nového snímku z kamery
# Příjme rozpoznané gesto a zda ruka je v záběru
    def update(self, gesture: str | None, hand_present: bool) -> tuple[bool, float]:
        """
        Aktualizuje stav zámku.
        Vrací: (je_odemčeno, progres_odemykání_0_až_1)
        """
        now = time.time()

        # 1. Kontrola automatického uzamčení při delší nepřítomnosti ruky
        if hand_present:
            self.last_hand_seen_time = now
        else:
            self.arm_start_time = None
            if self.is_armed and (now - self.last_hand_seen_time) > self.auto_disarm_timeout:
                self.is_armed = False
            return self.is_armed, 0.0

        # Ochranná prodleva těsně po přepnutí stavu, aby se hned ne přepnul zpět
        if now < self.toggle_cooldown_until:
            return self.is_armed, 0.0

        # 2. Vyhodnocení držení aktivačního gesta (ARM_GESTURE)
        if gesture == self.arm_gesture:
            if self.arm_start_time is None:
                self.arm_start_time = now

            elapsed = now - self.arm_start_time
            progress = min(elapsed / self.hold_seconds, 1.0)

            if progress >= 1.0:
                self.is_armed = not self.is_armed
                self.arm_start_time = None
                self.toggle_cooldown_until = now + 1.0  # 1s pauza proti nechtěnému přepnutí zpět
                return self.is_armed, 0.0

            return self.is_armed, progress
        else:
            self.arm_start_time = None
            return self.is_armed, 0.0

#  Vykreslí do snímku z kamery vizuální indikátor stavu
#  Abychom viděli co pojistka dělá, např je ARMED
    def draw_hud(self, frame: np.ndarray, progress: float) -> None:
        h, w, _ = frame.shape
        now = time.time()

        if self.is_armed:
            remaining = max(0.0, self.auto_disarm_timeout - (now - self.last_hand_seen_time))
            status_text = (
                f"ARMED (Auto-lock: {remaining:.1f}s)"
                if remaining < self.auto_disarm_timeout
                else "ARMED [AKTIVNI]"
            )
            color = (0, 255, 0)
        else:
            status_text = f"DISARMED (Drz {self.arm_gesture} {self.hold_seconds:.1f}s)"
            color = (0, 140, 255)

        cv2.putText(
            frame,
            status_text,
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

        if progress > 0.0:
            bar_x, bar_y, bar_w, bar_h = 20, 82, 260, 12
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
            fill_w = int(bar_w * progress)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 255, 255), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)