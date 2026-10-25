"""Scenario vocabularies: what each kind of death looks like, in pt-BR forensic register.

Every string here is latin-1 encodable (fpdf2 core fonts). Places carry their preposition ("no quarto"). Histories are `string.Template`
texts: `$place`, `$position`, `$time` and `$substance` are filled by the realizer and
`{{DEATH_DATE}}` is a draft slot.
"""

from dataclasses import dataclass
from typing import get_args

from necrohub.contracts.gold import Scenario

SCENARIOS: tuple[Scenario, ...] = get_args(Scenario)

# (city, UF, phone area code). Real cities, generic institute names: no real people.
CITIES: tuple[tuple[str, str, str], ...] = (
    ("São Luís", "MA", "98"),
    ("Imperatriz", "MA", "99"),
    ("Teresina", "PI", "86"),
    ("Fortaleza", "CE", "85"),
    ("Belém", "PA", "91"),
    ("Recife", "PE", "81"),
    ("Natal", "RN", "84"),
    ("Salvador", "BA", "71"),
    ("Goiânia", "GO", "62"),
    ("Campinas", "SP", "19"),
    ("Curitiba", "PR", "41"),
    ("Belo Horizonte", "MG", "31"),
    ("Palmas", "TO", "63"),
    ("Maceió", "AL", "82"),
    ("Aracaju", "SE", "79"),
    ("João Pessoa", "PB", "83"),
)
INSTITUTES = (
    "Instituto Médico-Legal",
    "Instituto Médico-Legal Regional",
    "Instituto de Medicina Legal",
)
STREET_TYPES = ("Rua", "Rua", "Avenida", "Travessa", "Alameda")
NEIGHBOURHOODS = (
    "Centro",
    "Vila Nova",
    "Jardim América",
    "Bairro Novo",
    "Jardim das Flores",
    "Vila Esperança",
    "Parque das Árvores",
    "Alto da Colina",
    "Residencial Primavera",
    "Conjunto Bela Vista",
)
SKIN_COLORS = ("branca", "parda", "parda", "preta", "amarela")
MONTHS = (
    "janeiro",
    "fevereiro",
    "março",
    "abril",
    "maio",
    "junho",
    "julho",
    "agosto",
    "setembro",
    "outubro",
    "novembro",
    "dezembro",
)
SUBSTANCES = ("etanol", "cocaína", "carbamato", "organofosforado")

ORGANS = ("Encéfalo", "Pescoço", "Pulmões", "Coração", "Fígado", "Baço", "Rins", "Estômago")
OPTIONAL_ORGANS = frozenset({"Pescoço"})  # described only when a variant finds something there
NORMAL_FINDINGS = (
    "sem alterações macroscópicas",
    "de forma, volume e consistência habituais",
    "sem alterações macroscópicas dignas de nota",
    "de aspecto habitual ao exame macroscópico",
)


@dataclass(frozen=True)
class PostmortemState:
    hours: tuple[int, int]
    rigor: str
    livor: str
    decomposition: str


STATES = (
    PostmortemState(
        (2, 6),
        "parcial, em mandíbula e membros superiores",
        "não fixos, em regiões de declive",
        "sem sinais de putrefação",
    ),
    PostmortemState(
        (8, 12), "generalizada", "fixos, em regiões de declive", "sem sinais de putrefação"
    ),
    PostmortemState(
        (12, 24),
        "generalizada",
        "fixos, em regiões de declive",
        "mancha verde abdominal incipiente",
    ),
    PostmortemState(
        (24, 36),
        "em resolução",
        "fixos, violáceos",
        "mancha verde abdominal e circulação póstuma",
    ),
    PostmortemState(
        (72, 120),
        "ausente",
        "de difícil caracterização",
        "fase gasosa, com enfisema putrefativo e descolamento da epiderme",
    ),
)


@dataclass(frozen=True)
class InjurySpec:
    kind: str
    regions: tuple[str, ...]
    sides: tuple[str | None, ...]
    size: tuple[float, float] | None
    features: tuple[str, ...]


@dataclass(frozen=True)
class Variant:
    """One coherent cause of death: the injuries that produce it, the organ findings they
    explain, its mechanism and extra exams. Non-empty `places` or `manners` replace the
    scenario's (a fall from height needs a height; an overdose is rarely a suicide)."""

    cause: str
    mechanism: str
    findings: dict[str, str]
    exams: tuple[str, ...] = ()
    injuries: tuple[InjurySpec, ...] = ()
    places: tuple[str, ...] = ()
    manners: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScenarioSpec:
    """`injuries` are present in every case of the scenario but cause nothing on their own
    (maceration after drowning); `optional_injuries`
    are incidental (defence wounds, abrasions from falling to the ground)."""

    places: tuple[str, ...]
    positions: tuple[str, ...]
    histories: tuple[str, ...]
    variants: tuple[Variant, ...]
    manners: tuple[str, ...]
    exams: tuple[str, ...]
    states: tuple[int, ...]
    ages: tuple[int, int]
    male_share: float
    injuries: tuple[InjurySpec, ...] = ()
    optional_injuries: tuple[InjurySpec, ...] = ()


_DECUBITUS = ("decúbito dorsal", "decúbito ventral", "decúbito lateral direito")
_ETHANOL = "dosagem de etanol no sangue"
_ABRASION = InjurySpec(
    "escoriações",
    ("face anterior do joelho", "região frontal", "face lateral do braço"),
    ("direita", "esquerda", None),
    (1.0, 6.0),
    ("crosta hemática", "forma irregular", "aspecto apergaminhado"),
)
_ENTRY_FEATURES = (
    "orla de escoriação",
    "orla de enxugo",
    "zona de tatuagem e esfumaçamento",
    "bordas invertidas",
)
_EXIT = InjurySpec(
    "ferimento perfurocontuso de saída",
    ("região dorsal", "região escapular"),
    ("esquerda",),
    (1.2, 3.5),
    ("bordas evertidas e irregulares", "ausência de orla de escoriação"),
)
_STAB_FEATURES = (
    "bordas regulares e lisas",
    "ângulos agudos",
    "cauda de escoriação em um dos ângulos",
    "fundo com infiltração hemorrágica",
)
_HEAD_WOUND = InjurySpec(
    "ferida contusa",
    ("região frontoparietal", "região occipital", "região parietal"),
    ("direita", "esquerda", None),
    (2.0, 7.0),
    ("bordas irregulares e escoriadas", "pontes de tecido no fundo da lesão"),
)
_TRUNK_BRUISE = InjurySpec(
    "equimose",
    ("região torácica lateral", "hipocôndrio", "flanco"),
    ("direita", "esquerda"),
    (4.0, 12.0),
    ("coloração arroxeada", "forma irregular"),
)
_TRUNK_ABRASION = InjurySpec(
    "escoriações",
    ("região torácica lateral", "região dorsal", "flanco"),
    ("direita", "esquerda"),
    (3.0, 15.0),
    ("aspecto apergaminhado", "crosta hemática", "disposição em placa"),
)

SPECS: dict[Scenario, ScenarioSpec] = {
    "firearm_homicide": ScenarioSpec(
        places=("em via pública", "em um terreno baldio", "na calçada em frente a uma residência"),
        positions=_DECUBITUS,
        histories=(
            "Segundo a guia policial, a vítima foi encontrada $place, em $position, após relato de disparos de arma de fogo por volta das $time do dia {{DEATH_DATE}}.",
            "Conforme informações policiais, a vítima foi atingida por disparos de arma de fogo $place e encontrada em $position. O óbito teria ocorrido no local, por volta das $time de {{DEATH_DATE}}.",
        ),
        variants=(
            Variant(
                "Choque hipovolêmico secundário a ferimento cardíaco por projétil de arma de fogo",
                "hemorragia interna aguda por ação perfurocontundente",
                {
                    "Coração": "lesão transfixante do ventrículo esquerdo, com hemopericárdio",
                    "Pulmões": "lobo superior do pulmão esquerdo transfixado pelo trajeto do projétil, com hemotórax esquerdo de cerca de 1.200 mL",
                },
                injuries=(
                    InjurySpec(
                        "ferimento perfurocontuso de entrada",
                        ("região precordial", "região torácica anterior"),
                        ("esquerda",),
                        (0.7, 1.2),
                        _ENTRY_FEATURES,
                    ),
                    _EXIT,
                ),
            ),
            Variant(
                "Hemorragia interna aguda por lesão pulmonar causada por projétil de arma de fogo",
                "choque hipovolêmico por ação perfurocontundente",
                {
                    "Pulmões": "laceração do lobo inferior do pulmão esquerdo, com hemotórax esquerdo de cerca de 1.800 mL"
                },
                injuries=(
                    InjurySpec(
                        "ferimento perfurocontuso de entrada",
                        ("região torácica anterior", "região torácica lateral"),
                        ("esquerda",),
                        (0.7, 1.2),
                        _ENTRY_FEATURES,
                    ),
                    _EXIT,
                ),
            ),
        ),
        manners=("violenta (homicídio)",),
        exams=(
            _ETHANOL,
            "radiografia de tórax para pesquisa de fragmentos de projétil",
            "exame residuográfico das mãos",
        ),
        states=(0, 1, 2),
        ages=(16, 55),
        male_share=0.85,
        optional_injuries=(_ABRASION,),
    ),
    "sharp_force": ScenarioSpec(
        places=("no interior de uma residência", "em via pública", "no quintal de uma residência"),
        positions=_DECUBITUS,
        histories=(
            "Segundo o histórico policial, a vítima foi golpeada com instrumento perfurocortante durante uma briga $place, sendo encontrada em $position por volta das $time de {{DEATH_DATE}}.",
            "Consta da requisição que a vítima foi agredida com faca $place e encontrada em $position. O óbito ocorreu por volta das $time do dia {{DEATH_DATE}}.",
        ),
        variants=(
            Variant(
                "Choque hipovolêmico por ferimentos produzidos por instrumento perfurocortante",
                "hemorragia aguda interna e externa",
                {
                    "Fígado": "laceração do lobo esquerdo, com hemoperitônio de cerca de 1.000 mL",
                    "Estômago": "perfuração da parede anterior",
                },
                injuries=(
                    InjurySpec(
                        "ferida perfuroincisa",
                        ("região epigástrica", "hipocôndrio", "flanco", "região mesogástrica"),
                        ("esquerda", None),
                        (1.5, 3.5),
                        _STAB_FEATURES,
                    ),
                ),
            ),
            Variant(
                "Tamponamento cardíaco por ferimento produzido por instrumento perfurocortante",
                "compressão cardíaca por hemopericárdio",
                {"Coração": "ferida do ventrículo direito, com hemopericárdio de cerca de 350 mL"},
                injuries=(
                    InjurySpec(
                        "ferida perfuroincisa",
                        ("região precordial", "região torácica anterior", "região epigástrica"),
                        ("esquerda", None),
                        (1.5, 3.5),
                        _STAB_FEATURES,
                    ),
                ),
            ),
        ),
        manners=("violenta (homicídio)",),
        exams=(_ETHANOL, "exame histopatológico das bordas das feridas"),
        states=(0, 1, 2),
        ages=(16, 60),
        male_share=0.75,
        optional_injuries=(
            InjurySpec(
                "ferida incisa",
                ("face palmar da mão",),
                ("direita", "esquerda"),
                (1.0, 4.0),
                ("aspecto superficial, compatível com lesão de defesa", "bordas regulares"),
            ),
        ),
    ),
    "traffic": ScenarioSpec(
        places=(
            "em uma rodovia estadual",
            "no cruzamento de uma avenida movimentada",
            "no acostamento de uma rodovia federal",
        ),
        positions=_DECUBITUS,
        histories=(
            "De acordo com o boletim de ocorrência, a vítima, motociclista, colidiu com um automóvel $place por volta das $time de {{DEATH_DATE}} e foi encontrada em $position, sem sinais vitais.",
            "A vítima, pedestre, foi atropelada $place por volta das $time de {{DEATH_DATE}}. Foi atendida pelo serviço móvel de urgência, mas evoluiu a óbito no local, em $position.",
        ),
        variants=(
            Variant(
                "Traumatismo cranioencefálico",
                "ação contundente de alta energia",
                {
                    "Encéfalo": "hemorragia subaracnóidea difusa e focos de contusão cortical, com fratura da calota craniana"
                },
                injuries=(_HEAD_WOUND,),
            ),
            Variant(
                "Politraumatismo com choque hemorrágico",
                "ação contundente de alta energia com hemorragia interna",
                {
                    "Baço": "ruptura do polo inferior, com hemoperitônio",
                    "Fígado": "lacerações múltiplas do lobo direito",
                    "Pulmões": "contusão pulmonar bilateral",
                },
                injuries=(_TRUNK_BRUISE, _TRUNK_ABRASION),
            ),
        ),
        manners=("violenta (acidente de trânsito)",),
        exams=(_ETHANOL, "pesquisa de cocaína e metabólitos na urina"),
        states=(0, 1),
        ages=(18, 70),
        male_share=0.8,
        optional_injuries=(
            InjurySpec(
                "escoriações",
                ("face lateral do braço", "face anterior da coxa", "região malar"),
                ("direita", "esquerda"),
                (3.0, 15.0),
                ("aspecto apergaminhado", "crosta hemática", "disposição em placa"),
            ),
            InjurySpec(
                "fratura exposta",
                ("terço médio da perna",),
                ("direita", "esquerda"),
                None,
                ("exposição de fragmentos ósseos da tíbia",),
            ),
        ),
    ),
    "fall": ScenarioSpec(
        places=(
            "do terraço de um edifício residencial",
            "de um andaime em um canteiro de obras",
            "na escadaria de uma residência",
        ),
        positions=_DECUBITUS,
        histories=(
            "Segundo relato de testemunhas registrado na guia policial, a vítima sofreu queda $place por volta das $time de {{DEATH_DATE}}, sendo encontrada em $position.",
            "Informações da equipe de remoção indicam queda acidental $place. A vítima foi encontrada em $position, e o óbito teria ocorrido por volta das $time do dia {{DEATH_DATE}}.",
        ),
        variants=(
            Variant(
                "Traumatismo cranioencefálico por queda",
                "ação contundente com hemorragia intracraniana",
                {
                    "Encéfalo": "hematoma subdural agudo, com desvio da linha média e fratura linear do osso occipital"
                },
                injuries=(
                    InjurySpec(
                        "ferida contusa",
                        ("região occipital", "região parieto-occipital"),
                        ("direita", "esquerda", None),
                        (2.0, 7.0),
                        ("bordas irregulares e escoriadas", "pontes de tecido no fundo da lesão"),
                    ),
                ),
            ),
            Variant(
                "Politraumatismo por precipitação",
                "ação contundente com hemorragia interna",
                {
                    "Fígado": "lacerações da face diafragmática, com hemoperitônio",
                    "Pulmões": "contusões e lacerações pulmonares, com hemotórax bilateral",
                },
                injuries=(_TRUNK_BRUISE, _TRUNK_ABRASION),
                places=(
                    "do terraço de um edifício residencial",
                    "de um andaime em um canteiro de obras",
                ),
            ),
        ),
        manners=("violenta (acidente)",),
        exams=(_ETHANOL, "exame histopatológico de fragmentos do encéfalo"),
        states=(0, 1, 2),
        ages=(20, 85),
        male_share=0.65,
        optional_injuries=(_ABRASION,),
    ),
    "hanging": ScenarioSpec(
        places=(
            "no quarto da residência",
            "na área de serviço da residência",
            "em uma árvore de um terreno próximo à residência",
        ),
        positions=(
            "em suspensão completa, com laço de corda ao redor do pescoço",
            "em suspensão incompleta, com os pés tocando o solo e laço de tecido ao redor do pescoço",
        ),
        histories=(
            "Familiares encontraram a vítima $place, $position, por volta das $time de {{DEATH_DATE}}. A equipe de socorro constatou o óbito no local.",
            "Conforme a guia policial, a vítima foi localizada $place, $position. Não havia sinais de arrombamento. O óbito teria ocorrido por volta das $time do dia {{DEATH_DATE}}.",
        ),
        variants=(
            Variant(
                "Asfixia mecânica por enforcamento",
                "constrição do pescoço por laço acionado pelo peso do próprio corpo",
                {
                    "Pescoço": "infiltração hemorrágica dos músculos esternocleidomastóideos e fratura do corno maior esquerdo do osso hioide",
                    "Pulmões": "congestão e edema, com petéquias subpleurais",
                    "Coração": "petéquias subepicárdicas",
                },
                injuries=(
                    InjurySpec(
                        "sulco de enforcamento",
                        ("região cervical anterior e lateral",),
                        (None,),
                        (0.5, 1.5),
                        (
                            "trajeto oblíquo e ascendente em direção à nuca",
                            "interrupção na região occipital, no ponto do nó",
                            "fundo apergaminhado",
                        ),
                    ),
                ),
            ),
        ),
        manners=("violenta (suicídio)", "violenta (compatível com suicídio)"),
        exams=(_ETHANOL, "exame histopatológico do sulco cervical"),
        states=(0, 1, 2),
        ages=(16, 75),
        male_share=0.75,
    ),
    "drowning": ScenarioSpec(
        places=(
            "na margem de um rio",
            "em uma praia urbana",
            "em um açude de uma propriedade rural",
        ),
        positions=("decúbito ventral, parcialmente submerso", "decúbito dorsal"),
        histories=(
            "O corpo foi retirado da água $place pelo Corpo de Bombeiros e encontrava-se em $position. Testemunhas relatam que a vítima entrou na água por volta das $time de {{DEATH_DATE}} e não retornou.",
            "Segundo a guia policial, a vítima desapareceu durante banho $place por volta das $time do dia {{DEATH_DATE}}. O corpo foi localizado horas depois, em $position.",
        ),
        variants=(
            Variant(
                "Asfixia mecânica por afogamento",
                "aspiração de meio líquido com obstrução das vias aéreas",
                {
                    "Pulmões": "aumentados de volume, com enfisema aquoso e líquido espumoso na traqueia e nos brônquios",
                    "Estômago": "grande quantidade de líquido aquoso misturado a areia",
                },
            ),
        ),
        manners=("violenta (acidente)",),
        exams=(_ETHANOL, "pesquisa de diatomáceas em medula óssea"),
        states=(1, 2, 3),
        ages=(18, 70),  # adults only: the body measurements are adult ones
        male_share=0.8,
        injuries=(
            InjurySpec(
                "maceração da epiderme",
                ("palmas das mãos e plantas dos pés",),
                (None,),
                None,
                ("aspecto de mãos de lavadeira",),
            ),
        ),
        optional_injuries=(
            InjurySpec(
                "escoriações",
                ("região frontal", "face dorsal da mão"),
                ("direita", "esquerda", None),
                (1.0, 5.0),
                ("ausência de reação vital, compatível com arrasto",),
            ),
        ),
    ),
    "cardiac_natural": ScenarioSpec(
        places=(
            "no quarto da residência",
            "no local de trabalho",
            "em via pública, próximo a uma parada de ônibus",
        ),
        positions=_DECUBITUS,
        histories=(
            "Segundo familiares, a vítima tinha diagnóstico de hipertensão arterial e queixou-se de dor torácica antes de ser encontrada $place, em $position. O óbito teria ocorrido por volta das $time de {{DEATH_DATE}}.",
            "A vítima apresentou mal súbito $place por volta das $time do dia {{DEATH_DATE}} e foi encontrada em $position. Não há notícia de violência.",
        ),
        variants=(
            Variant(
                "Infarto agudo do miocárdio",
                "insuficiência cardíaca aguda por oclusão coronariana",
                {
                    "Coração": "hipertrofia ventricular esquerda e placa aterosclerótica com estenose de cerca de 80 % da artéria descendente anterior, com área de palidez e amolecimento na parede anterior do ventrículo esquerdo",
                    "Pulmões": "congestão e edema",
                },
                ("exame histopatológico do miocárdio",),
            ),
            Variant(
                "Cardiopatia hipertensiva",
                "arritmia cardíaca",
                {
                    "Coração": "cardiomegalia, com hipertrofia concêntrica do ventrículo esquerdo",
                    "Rins": "superfície granular, compatível com nefroesclerose hipertensiva",
                },
                ("exame histopatológico do miocárdio e dos rins",),
            ),
        ),
        manners=("natural",),
        exams=(_ETHANOL,),
        states=(0, 1, 2),
        ages=(40, 88),
        male_share=0.6,
    ),
    "other_natural": ScenarioSpec(
        places=("na própria residência", "na casa de familiares"),
        positions=_DECUBITUS,
        histories=(
            "Familiares relatam que a vítima tinha hipertensão arterial e outras doenças crônicas e foi encontrada sem vida $place, em $position, por volta das $time de {{DEATH_DATE}}. Não há notícia de violência.",
            "A vítima, com histórico de doenças crônicas, foi encontrada $place, em $position. O óbito teria ocorrido por volta das $time do dia {{DEATH_DATE}}, sem assistência médica.",
        ),
        variants=(
            Variant(
                "Acidente vascular encefálico hemorrágico",
                "hipertensão intracraniana",
                {
                    "Encéfalo": "hematoma intraparenquimatoso nos núcleos da base à esquerda, com inundação ventricular"
                },
            ),
            Variant(
                "Tromboembolismo pulmonar",
                "insuficiência cardiorrespiratória aguda",
                {
                    "Pulmões": "trombo no tronco da artéria pulmonar, estendendo-se aos ramos principais"
                },
            ),
            Variant(
                "Broncopneumonia",
                "insuficiência respiratória aguda",
                {
                    "Pulmões": "áreas de consolidação bilateral, com secreção purulenta nos brônquios"
                },
            ),
        ),
        manners=("natural",),
        exams=(_ETHANOL, "exame histopatológico dos pulmões e do encéfalo"),
        states=(0, 1, 2),
        ages=(35, 90),
        male_share=0.5,
    ),
    "poisoning": ScenarioSpec(
        places=(
            "no quarto da residência",
            "na cozinha da residência",
            "no galpão de uma propriedade rural",
        ),
        positions=_DECUBITUS,
        histories=(
            "Familiares encontraram a vítima $place, em $position, ao lado de uma embalagem contendo $substance. O óbito teria ocorrido por volta das $time de {{DEATH_DATE}}.",
            "Segundo a guia policial, a vítima foi encontrada $place, em $position, e havia no local indícios de uso de $substance. O óbito teria ocorrido por volta das $time do dia {{DEATH_DATE}}.",
        ),
        variants=(
            Variant(
                "Intoxicação exógena por carbamato",
                "inibição da acetilcolinesterase com insuficiência respiratória",
                {
                    "Estômago": "conteúdo com grânulos escuros e odor característico",
                    "Pulmões": "edema pulmonar acentuado, com secreção abundante nas vias aéreas",
                },
                ("pesquisa de carbamato no conteúdo gástrico e no sangue",),
            ),
            Variant(
                "Intoxicação exógena por organofosforado",
                "crise colinérgica com insuficiência respiratória",
                {
                    "Estômago": "conteúdo líquido de odor aliáceo",
                    "Pulmões": "congestão e edema, com secreção abundante nas vias aéreas",
                },
                ("pesquisa de organofosforado no conteúdo gástrico e no sangue",),
            ),
            Variant(
                "Intoxicação aguda por cocaína",
                "arritmia cardíaca",
                {
                    "Coração": "hipertrofia ventricular esquerda, sem oclusão coronariana",
                    "Pulmões": "congestão e edema pulmonar",
                },
                ("pesquisa de cocaína e metabólitos no sangue e na urina",),
                places=("no quarto da residência", "em um imóvel abandonado"),
                manners=("violenta (acidente)", "indeterminada"),
            ),
        ),
        manners=("violenta (suicídio)", "violenta (acidente)", "indeterminada"),
        exams=(_ETHANOL, "exame histopatológico do fígado e dos rins"),
        states=(0, 1, 2),
        ages=(16, 70),
        male_share=0.6,
    ),
    "undetermined": ScenarioSpec(
        places=("em um matagal às margens de uma estrada vicinal", "em um imóvel abandonado"),
        positions=("decúbito dorsal", "decúbito ventral"),
        histories=(
            "O corpo foi localizado $place, em $position, em avançado estado de decomposição, após denúncia de odor fétido. Não há testemunhas; estima-se que o óbito tenha ocorrido em {{DEATH_DATE}}, por volta das $time.",
        ),
        variants=(
            Variant(
                "Indeterminada, em razão do avançado estado de putrefação",
                "indeterminado",
                {
                    "Encéfalo": "liquefeito pela putrefação",
                    "Pulmões": "em avançado estado de autólise",
                    "Coração": "em avançado estado de autólise, sem lesões identificáveis",
                    "Fígado": "em avançado estado de autólise",
                },
            ),
        ),
        manners=("indeterminada",),
        exams=(
            "exame toxicológico em fragmentos de fígado",
            "exame odontolegal e papiloscópico para confirmação da identidade",
        ),
        states=(4,),
        ages=(20, 80),
        male_share=0.7,
    ),
}


def variant_of(scenario: Scenario, cause: str) -> Variant:
    """The variant a case was sampled from (causes are unique within a scenario)."""
    return next(v for v in SPECS[scenario].variants if v.cause == cause)
