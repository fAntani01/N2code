"""
Riquadro di stato del dispositivo (cmn_group_overview): mostra i voltaggi
correnti applicati a ogni MEMS e a quale antenna/porta corrisponde ogni
parametro di scattering, sullo schema semplificato del layout del chip.

Le posizioni sono definite tramite due dizionari (MEMS_POSITIONS,
TRACE_POSITIONS): modifica SOLO quelli per adattare lo schema al layout
reale del dispositivo — la logica di disegno in DeviceOverviewWidget non
va toccata per un semplice riposizionamento.

Disegnato con QPainter (rettangoli + testo): nessuna libreria aggiuntiva,
coerente con lo stile "widget nativo" gia' usato nel resto della GUI.
"""

from typing import Dict, Optional, Tuple

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QPainter, QColor, QPen, QFont
from PyQt6.QtWidgets import QWidget

# ----------------------------------------------------------------------
# LAYOUT DEL DEVICE (da adattare al chip reale)
# ----------------------------------------------------------------------
# Posizione (x, y) di ciascun MEMS sullo schema, in coordinate relative
# 0-1 (frazione di larghezza/altezza del widget, origine in alto a
# sinistra). Un MEMS per canale DAQ, disposti dall'alto verso il basso
# come i "denti" del device (vedi immagine di riferimento).
# TODO: aggiusta i valori (e aggiungi/togli canali) secondo il layout
# reale e il numero di canali effettivamente cablati.
MEMS_POSITIONS: Dict[int, Tuple[float, float]] = {
    0: (0.45, 0.2),
    1: (0.45, 0.40),
    2: (0.45, 0.60),
    3: (0.45, 0.80),
    4: (0.65, 0.20),
    5: (0.65, 0.40),
    6: (0.65, 0.60),
    7: (0.65, 0.80),
}

# Posizione + lato di ciascuna antenna/parametro S. "side" determina solo
# l'allineamento del testo (per non sovrapporlo allo schema): "left" per
# le antenne di output (a sinistra del device), "right" per l'antenna di
# input (a destra).
TRACE_POSITIONS: Dict[str, dict] = {
    "S21": {"pos": (0.92, 0.15), "side": "right"},
    "S31": {"pos": (0.92, 0.50), "side": "right"},
    "S41": {"pos": (0.92, 0.85), "side": "right"},
    "S11": {"pos": (0.05, 0.50), "side": "left"},
}

_MEMS_COLOR_MIN = QColor(90, 90, 90)  # grigio: voltaggio vicino a 0
_MEMS_COLOR_MAX = QColor(220, 60, 60)  # rosso: voltaggio al massimo della scala
_ANTENNA_COLOR = QColor(60, 200, 100)  # verde: antenne
_TEXT_COLOR = QColor(220, 220, 220)

# Scala (in modulo, V) usata per normalizzare il colore dei MEMS: un
# voltaggio con |V| >= questo valore e' disegnato al colore massimo
# (_MEMS_COLOR_MAX). Adatta al range reale dei tuoi voltaggi di pilotaggio
# (es. il daq_vmax configurato in Settings).
_MEMS_VOLTAGE_SCALE = 60.0


class DeviceOverviewWidget(QWidget):
    """Disegna lo schema del device con lo stato corrente dei MEMS
    (colore su un gradiente continuo grigio->rosso in base al voltaggio
    applicato, con il voltaggio scritto sopra) e le etichette dei
    parametri S sulle rispettive antenne (verde, fisse).

    Aggiornato chiamando update_state(voltages) ogni volta che i
    voltaggi applicati cambiano (Set Voltages, fine misura, ogni
    iterazione di training/search).
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumHeight(220)
        self._voltages: Dict[int, float] = {}

    def update_state(self, voltages: Dict[int, float]):
        """voltages: {canale: voltaggio applicato correntemente}. Canali
        assenti dal dizionario restano semplicemente non evidenziati
        (schema piu' piccolo del layout completo, o canale non attivo
        nella configurazione corrente)."""
        self._voltages = dict(voltages)
        self.update()  # richiede un repaint (chiama paintEvent)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        margin = 20
        w, h = self.width(), self.height()
        area = QRectF(margin, margin, max(w - 2 * margin, 1), max(h - 2 * margin, 1))

        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)

        self._draw_antennas(painter, area)
        self._draw_mems(painter, area)

    def _draw_antennas(self, painter: QPainter, area: QRectF):
        for trace, info in TRACE_POSITIONS.items():
            x, y = info["pos"]
            px = area.left() + x * area.width()
            py = area.top() + y * area.height()

            painter.setPen(QPen(_ANTENNA_COLOR, 2))
            painter.setBrush(_ANTENNA_COLOR)
            painter.drawEllipse(QRectF(px - 5, py - 5, 10, 10))

            is_right = info["side"] == "right"
            text_rect = QRectF(px - 68 if is_right else px + 8, py - 8, 60, 16)
            align = (Qt.AlignmentFlag.AlignRight if is_right else Qt.AlignmentFlag.AlignLeft) | Qt.AlignmentFlag.AlignVCenter
            painter.setPen(QPen(_ANTENNA_COLOR))
            painter.drawText(text_rect, align, trace)

    def _voltage_to_color(self, voltage: float) -> QColor:
        """Interpola tra _MEMS_COLOR_MIN e _MEMS_COLOR_MAX in base a
        |voltage| / _MEMS_VOLTAGE_SCALE, clampato a [0, 1]. Nessuna
        soglia di accensione: il colore rappresenta il voltaggio in modo
        continuo."""
        t = min(abs(voltage) / _MEMS_VOLTAGE_SCALE, 1.0) if _MEMS_VOLTAGE_SCALE > 0 else 0.0

        r = _MEMS_COLOR_MIN.red() + t * (_MEMS_COLOR_MAX.red() - _MEMS_COLOR_MIN.red())
        g = _MEMS_COLOR_MIN.green() + t * (_MEMS_COLOR_MAX.green() - _MEMS_COLOR_MIN.green())
        b = _MEMS_COLOR_MIN.blue() + t * (_MEMS_COLOR_MAX.blue() - _MEMS_COLOR_MIN.blue())
        return QColor(int(r), int(g), int(b))

    def _draw_mems(self, painter: QPainter, area: QRectF):
        for channel, (x, y) in MEMS_POSITIONS.items():
            if channel not in self._voltages:
                continue  # canale non presente nella configurazione corrente

            voltage = self._voltages[channel]
            color = self._voltage_to_color(voltage)

            px = area.left() + x * area.width()
            py = area.top() + y * area.height()

            painter.setPen(QPen(color, 2))
            painter.setBrush(color)
            painter.drawRect(QRectF(px - 6, py - 6, 12, 12))

            # Etichetta sopra al quadratino, centrata orizzontalmente.
            painter.setPen(QPen(_TEXT_COLOR))
            label_rect = QRectF(px - 40, py - 24, 80, 16)
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom, f"Ch{channel}: {voltage:.2f}V")