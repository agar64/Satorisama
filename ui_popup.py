from __future__ import annotations
from PySide6 import QtCore, QtGui, QtWidgets

class PopupWindow(QtWidgets.QFrame):
    def __init__(self) -> None:
        super().__init__(None)
        self.setWindowFlags(
            QtCore.Qt.WindowType.Tool
            | QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self._content = QtWidgets.QFrame()
        self._content.setObjectName("content")
        self._content.setStyleSheet("""
            QFrame#content {
                background: rgba(25, 25, 28, 235);
                border: 1px solid rgba(255, 255, 255, 40);
                border-radius: 10px;
            }
            QLabel {
                color: #f2f2f2;
                font-size: 13px;
            }
            QLabel#title {
                font-weight: 600;
                font-size: 12px;
                color: rgba(242, 242, 242, 180);
            }
        """)

        self._title = QtWidgets.QLabel("JP Hover OCR")
        self._title.setObjectName("title")
        self._body = QtWidgets.QLabel("")
        self._body.setWordWrap(True)
        self._body.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)

        layout = QtWidgets.QVBoxLayout(self._content)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(6)
        layout.addWidget(self._title)
        layout.addWidget(self._body)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._content)

    def set_text(self, lines: list[str]) -> None:
        if not lines:
            self._body.setText("(no text)")
        else:
            self._body.setText("\n".join(lines))
        self.adjustSize()

    def show_near(self, x: int, y: int) -> None:
        # Place to the right & slightly below cursor; clamp to screen
        x += 60
        y += 30
        screen = QtGui.QGuiApplication.screenAt(QtCore.QPoint(x, y)) or QtGui.QGuiApplication.primaryScreen()
        geo = screen.availableGeometry()

        margin = 12
        self.adjustSize()
        w = self.width()
        h = self.height()

        px = x + margin
        py = y + margin

        if px + w > geo.right():
            px = x - w - margin
        if py + h > geo.bottom():
            py = y - h - margin

        px = max(geo.left(), min(px, geo.right() - w))
        py = max(geo.top(), min(py, geo.bottom() - h))

        self.move(px, py)
        self.show()
        self.raise_()
        self.activateWindow()
