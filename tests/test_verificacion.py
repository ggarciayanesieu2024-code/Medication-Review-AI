
"""Tests for CIMA citation verification."""

from src.modelos import Cita, Fragmento
from src.verificacion import verificar_cita


def test_valid_quote():
    fragment = Fragmento(
        nregistro="123",
        seccion="4.5",
        texto="Este medicamento puede aumentar el efecto de otros medicamentos anticoagulantes.",
    )
    citation = Cita(
        nregistro="123",
        seccion="4.5",
        literal_quote="puede aumentar el efecto de otros medicamentos",
    )

    assert verificar_cita(citation, fragment) is True


def test_invented_quote():
    fragment = Fragmento(
        nregistro="123",
        seccion="4.5",
        texto="Este medicamento puede aumentar el efecto de otros medicamentos anticoagulantes.",
    )
    citation = Cita(
        nregistro="123",
        seccion="4.5",
        literal_quote="Este medicamento elimina completamente todos los riesgos",
    )

    assert verificar_cita(citation, fragment) is False


def test_short_quote():
    fragment = Fragmento(
        nregistro="123",
        seccion="4.5",
        texto="Este medicamento puede aumentar el efecto de otros medicamentos anticoagulantes.",
    )
    citation = Cita(
        nregistro="123",
        seccion="4.5",
        literal_quote="aumentar el efecto",
    )

    assert verificar_cita(citation, fragment) is False


def test_wrong_section():
    fragment = Fragmento(
        nregistro="123",
        seccion="4.5",
        texto="Este medicamento puede aumentar el efecto de otros medicamentos anticoagulantes.",
    )
    citation = Cita(
        nregistro="123",
        seccion="4.4",
        literal_quote="puede aumentar el efecto de otros medicamentos",
    )

    assert verificar_cita(citation, fragment) is False
