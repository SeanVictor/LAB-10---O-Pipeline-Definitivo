"""
Passo 2: Simulando o RAG Massivo

Gera um texto ficticio de aproximadamente 10.000 a 15.000 tokens
simulando os PDFs medicos recuperados pelo banco vetorial do Lab 09.
Passa o texto pelo AutoTokenizer do modelo para validar o tamanho.

Aluno      : Sean Victor Machado de Moraes
GitHub     : https://github.com/SeanVictor
Disciplina : Topicos em IA - 2026.1
Instituicao: iCEV
"""


# ─────────────────────────────────────────────────────────────
# Corpus medico simulado (5 capitulos de manuais clinicos)
# ─────────────────────────────────────────────────────────────

CAPITULO_1 = """
CAPITULO 1 - FUNDAMENTOS DE CARDIOLOGIA CLINICA

1.1 Anatomia Funcional do Coracao
O coracao e um orgao muscular oco situado no mediastino medio, levemente desviado para a esquerda
da linha mediana. Seu peso medio varia entre 250 e 350 gramas no adulto saudavel, podendo aumentar
significativamente em condicoes patologicas como hipertrofia ventricular esquerda ou cardiomiopatia
dilatada. A parede cardiaca e composta por tres camadas distintas: o epicardio, o miocardio e o
endocardio. O miocardio constitui a maior parte da massa cardiaca e e responsavel pela funcao
contrutil do orgao.

1.2 Fisiologia da Conducao Eletrica
O sistema de conducao cardiaca comeca no no sinoatrial (SA), localizado na juncao da veia cava
superior com o atrio direito. O no SA gera impulsos eletricas a uma frequencia de 60 a 100 batimentos
por minuto em condicoes de repouso. Esses impulsos se propagam pelos atrios ate alcancar o no
atrioventricular (AV), que introduz um atraso fisiologico de aproximadamente 120 a 200 milissegundos.
Apos o no AV, o impulso percorre o feixe de His, os ramos direito e esquerdo e as fibras de Purkinje,
ativando os ventriculos de forma sincronizada e eficiente.

1.3 Insuficiencia Cardiaca Congestiva
A insuficiencia cardiaca congestiva (ICC) representa uma sindrome clinica complexa na qual o coracao
nao consegue bombear sangue suficiente para atender as demandas metabolicas dos tecidos. A ICC
pode ser classificada conforme a fracao de ejecao: ICC com fracao de ejecao reduzida (ICFEr), quando
a fEje e inferior a 40%, e ICC com fracao de ejecao preservada (ICFEp), quando a fEje e igual ou
superior a 50%. Os criterios de Framingham sao amplamente utilizados para o diagnostico clinico,
incluindo criterios maiores como cardiomegalia radiologica, edema pulmonar agudo e terceira bulha
cardiaca, e criterios menores como edema de membros inferiores, tosse noturna e dispneia aos
esforcos habituais.

1.4 Tratamento Farmacologico da ICC
O tratamento da insuficiencia cardiaca com fracao de ejecao reduzida baseia-se em quatro pilares
farmacologicos fundamentais: inibidores da enzima conversora de angiotensina (IECA) ou bloqueadores
do receptor de angiotensina-neprilisina (ARNI), betabloqueadores, antagonistas da aldosterona e
inibidores do cotransportador sodio-glicose tipo 2 (iSGLT2). A associacao sacubitril/valsartana
demonstrou superioridade sobre o enalapril na reducao de mortalidade cardiovascular e hospitalizacoes
por ICC no estudo PARADIGM-HF. Os betabloqueadores carvedilol, metoprolol succinato e bisoprolol
possuem beneficio comprovado em ensaios clinicos randomizados de grande porte.

1.5 Diagnostico por Imagem na Cardiologia
O ecocardiograma transtorácico permanece como o exame de imagem de primeira linha na avaliacao
cardiologica. Permite a quantificacao da fracao de ejecao, a avaliacao da funcao diastolica, a
estimativa das pressoes de enchimento e a identificacao de alteracoes estruturais como valvulopatias,
doenca pericardica e massas intracardiacas. A ressonancia magnetica cardiaca oferece resolucao
tecidual superior e e considerada o padrao ouro para a quantificacao de volumes e massas ventriculares.
A cintilografia miocardica com tecido de perfusao continua sendo valiosa na avaliacao da viabilidade
miocardica e na deteccao de isquemia reversivel.
""" * 4

CAPITULO_2 = """
CAPITULO 2 - PNEUMOLOGIA E DOENCAS RESPIRATORIAS

2.1 Fisiologia da Respiracao
A respiracao e um processo vital que envolve a troca gasosa entre o organismo e o meio ambiente.
O sistema respiratorio e composto pelas vias aereas superiores, incluindo nariz, faringe e laringe,
e pelas vias aereas inferiores, que compreendem a traqueia, os bronquios e os pulmoes. Os pulmoes
sao orgaos pareados localizados na cavidade toracica, separados pelo mediastino. O pulmao direito
possui tres lobos e o pulmao esquerdo possui dois lobos, esta diferenca anatomica se deve a presenca
do coracao no mediastino esquerdo.

2.2 Doenca Pulmonar Obstrutiva Cronica
A doenca pulmonar obstrutiva cronica (DPOC) e caracterizada por obstrucao do fluxo aereo persistente
e progressiva, geralmente associada a resposta inflamatoria cronica das vias aereas e do pulmao a
particulas ou gases nocivos, sendo o tabagismo o principal fator de risco. O diagnostico e confirmado
pela espirometria, com relacao VEF1/CVF pos-broncodilatador inferior a 0,70. A classificacao GOLD
divide a gravidade da obstrucao em quatro estagios: GOLD 1 (leve, VEF1 maior ou igual a 80% do
previsto), GOLD 2 (moderado, VEF1 entre 50 e 79%), GOLD 3 (grave, VEF1 entre 30 e 49%) e GOLD 4
(muito grave, VEF1 inferior a 30%).

2.3 Asma Bronquica
A asma e uma doenca inflamatoria cronica das vias aereas caracterizada por hiper-responsividade
bronquica e obstrucao variavel e reversivel ao fluxo aereo. A fisiopatologia envolve inflamacao
mediada principalmente por eosinofilos e linfocitos T helper tipo 2, com liberacao de interleucinas
como IL-4, IL-5 e IL-13. O tratamento de manutencao baseia-se em corticosteroides inalatorios,
associados a broncodilatadores de longa duracao quando necessario. O conceito de controle da asma
envolve a avaliacao dos sintomas, o uso de medicacao de alivio, as limitacoes das atividades e a
funcao pulmonar.

2.4 Pneumonia Adquirida na Comunidade
A pneumonia adquirida na comunidade (PAC) e uma infeccao aguda do parenquima pulmonar que acomete
individuos fora do ambiente hospitalar. O Streptococcus pneumoniae continua sendo o agente etiologico
mais frequente, respondendo por 30 a 40% dos casos de PAC que necessitam hospitalizacao. O escore
CURB-65 e amplamente utilizado para estratificacao de risco, atribuindo um ponto para cada um dos
seguintes criterios: confusao mental, ureia superior a 7 mmol/L, frequencia respiratoria igual ou
superior a 30 irpm, pressao arterial sistolica inferior a 90 mmHg ou diastolica igual ou inferior a
60 mmHg e idade igual ou superior a 65 anos.

2.5 Cancer de Pulmao
O cancer de pulmao representa a principal causa de morte por cancer no mundo, com aproximadamente
1,8 milhao de obitos anuais. O tabagismo e responsavel por cerca de 85% dos casos. Os tipos
histologicos mais frequentes sao o adenocarcinoma, o carcinoma de celulas escamosas e o carcinoma
de pequenas celulas. O rastreamento com tomografia computadorizada de baixa dose em individuos de
alto risco demonstrou reducao na mortalidade por cancer de pulmao no estudo NLST.
""" * 4

CAPITULO_3 = """
CAPITULO 3 - NEUROLOGIA CLINICA E DOENCAS DO SISTEMA NERVOSO

3.1 Anatomia do Sistema Nervoso Central
O sistema nervoso central compreende o encefalo e a medula espinal. O encefalo e dividido em
cerebro, cerebelo e tronco encefalico. O cerebro constitui a maior parte do encefalo e e responsavel
pelas funcoes cognitivas superiores, incluindo pensamento, memoria, linguagem e comportamento. O
cortex cerebral e organizado em quatro lobos: frontal, parietal, temporal e occipital, cada um com
funcoes especializadas e interconectadas.

3.2 Acidente Vascular Cerebral
O acidente vascular cerebral (AVC) e uma emergencia medica caracterizada pelo surgimento agudo de
deficit neurologico focal resultante de isquemia ou hemorragia cerebral. O AVC isquemico corresponde
a aproximadamente 85% de todos os casos e resulta da oclusao de uma arteria cerebral por trombo ou
embolo. O tratamento trombolítico com alteplase intravenosa e indicado nas primeiras 4,5 horas do
inicio dos sintomas em pacientes selecionados, enquanto a trombectomia mecanica estende a janela
terapeutica ate 24 horas em casos com penumbra isquemica identificada por neuroimagem avancada.

3.3 Doenca de Parkinson
A doenca de Parkinson e o segundo disturbio neurodegenerativo mais comum, afetando aproximadamente
1% da populacao acima dos 60 anos. A fisiopatologia envolve a degeneracao progressiva dos neuronios
dopaminergicos da substantia nigra pars compacta, levando a deficiencia dopaminergica no estriado.
A tétrade clinica classica inclui tremor de repouso, rigidez, bradicinesia e instabilidade postural.
O tratamento farmacologico baseia-se na reposicao dopaminergica com levodopa combinada a carbidopa,
alem de agonistas dopaminergicos, inibidores da MAO-B e inibidores da COMT.

3.4 Epilepsia
A epilepsia e definida como uma doenca cerebral caracterizada por predisposicao persistente a gerar
crises epilepticas e pelas consequencias neurobiologicas, cognitivas, psicologicas e sociais dessa
condicao. A classificacao das crises epilepticas pela Liga Internacional Contra a Epilepsia divide-as
em crises de inicio focal, crises de inicio generalizado e crises de inicio desconhecido. O tratamento
com drogas antiepilepticas visa o controle completo das crises sem efeitos adversos significativos.

3.5 Demencias
As demencias constituem uma sindrome caracterizada pelo declinio progressivo das funcoes cognitivas
de intensidade suficiente para interferir nas atividades de vida diaria. A doenca de Alzheimer e a
forma mais comum de demencia, correspondendo a 60 a 70% dos casos. O diagnostico baseia-se em
criterios clinicos associados a biomarcadores de liquido cefalorraquidiano ou neuroimagem funcional.
""" * 4

CAPITULO_4 = """
CAPITULO 4 - ENDOCRINOLOGIA E METABOLISMO

4.1 Diabetes Mellitus
O diabetes mellitus e um grupo de doencas metabolicas caracterizado por hiperglicemia cronica
resultante de defeitos na secrecao de insulina, na acao da insulina ou em ambos. O diabetes tipo 1
resulta da destruicao autoimune das celulas beta pancreaticas, levando a deficiencia absoluta de
insulina. O diabetes tipo 2 e caracterizado por resistencia periferica a insulina associada a
deficiencia progressiva da secrecao de insulina pelas celulas beta. Os criterios diagnosticos incluem
glicemia de jejum igual ou superior a 126 mg/dL, glicemia de duas horas no teste de tolerancia oral
a glicose igual ou superior a 200 mg/dL, hemoglobina glicada igual ou superior a 6,5% ou glicemia
ao acaso igual ou superior a 200 mg/dL com sintomas classicos.

4.2 Doencas da Tireoide
A glandula tireoide e o maior orgao endocrino do corpo humano, localizada na regiao anterior do
pescoco. Produz os hormonios tireoidianos triiodotironina (T3) e tiroxina (T4), essenciais para o
metabolismo energetico, o crescimento e o desenvolvimento. O hipotireoidismo e caracterizado pela
producao insuficiente de hormonios tireoidianos e manifesta-se clinicamente por fadiga, ganho de
peso, intolerancia ao frio, constipacao e bradicardia. O hipertireoidismo resulta do excesso de
hormonios tireoidianos e apresenta-se com perda de peso, taquicardia, intolerancia ao calor e
nervosismo.

4.3 Obesidade
A obesidade e definida como acumulo excessivo de gordura corporal que representa risco a saude,
diagnosticada pelo indice de massa corporal (IMC) igual ou superior a 30 kg/m2. Constitui um dos
maiores problemas de saude publica do seculo XXI, com prevalencia estimada de 650 milhoes de adultos
afetados mundialmente. A fisiopatologia e multifatorial, envolvendo fatores geneticos, ambientais,
comportamentais, neuroendocrinos e psicologicos. O tratamento inclui modificacao do estilo de vida,
farmacoterapia com agentes como semaglutida e liraglutida, e cirurgia bariatrica para casos de
obesidade grave com comorbidades.
""" * 4

CAPITULO_5 = """
CAPITULO 5 - FARMACOLOGIA CLINICA E INTERACOES MEDICAMENTOSAS

5.1 Principios de Farmacocinetica
A farmacocinetica descreve o que o organismo faz com o farmaco, englobando os processos de
absorcao, distribuicao, metabolismo e excrecao. A absorcao e influenciada pela via de administracao,
pela solubilidade do farmaco, pelo pH gastrico e pela motilidade intestinal. A biodisponibilidade
representa a fracao do farmaco administrado que alcanca a circulacao sistemica de forma inalterada.
A distribuicao depende da ligacao as proteinas plasmaticas, da lipossolubilidade e do volume de
distribuicao aparente. O metabolismo ocorre predominantemente no figado pelas enzimas do citocromo
P450, podendo resultar em ativacao ou inativacao do farmaco.

5.2 Antibioticoterapia
O uso racional de antibioticos e fundamental para preservar a eficacia terapeutica e prevenir o
desenvolvimento de resistencia bacteriana. A escolha do antibiotico deve considerar o agente
etiologico provavel, o perfil de sensibilidade local, a farmacocinetica e farmacodinamica do farmaco,
as comorbidades do paciente e o custo do tratamento. Os beta-lactamicos, incluindo penicilinas,
cefalosporinas e carbapenens, atuam inibindo a sintese da parede celular bacteriana. As fluoroquinolonas
inibem a DNA girase e a topoisomerase IV, enquanto os macrolideos inibem a sintese proteica bacteriana
ao se ligar a subunidade 50S do ribossomo.

5.3 Interacoes Medicamentosas Relevantes
As interacoes medicamentosas ocorrem quando o efeito de um farmaco e modificado pela presenca de
outro farmaco, alimento, bebida ou suplemento. As interacoes farmacocineticas envolvem alteracoes
nos processos de absorcao, distribuicao, metabolismo ou excrecao. A inibicao do CYP3A4 pela
claritromicina pode elevar os niveis plasmaticos de estatinas como sinvastatina e atorvastatina,
aumentando o risco de miopatia. A rifampicina, potente indutor enzimatico, pode reduzir drasticamente
os niveis de anticoagulantes orais, imunossupressores e antiretrovirais, comprometendo a eficacia
terapeutica.
""" * 4

CORPUS_MEDICO = "\n\n".join([
    CAPITULO_1, CAPITULO_2, CAPITULO_3, CAPITULO_4, CAPITULO_5
])


def generate_rag_context(tokenizer, target_tokens: int = 12000) -> dict:
    """
    Gera o contexto massivo simulando o RAG e o tokeniza.

    Parametros:
        tokenizer    : tokenizador do modelo
        target_tokens: numero alvo de tokens

    Retorna:
        dict com:
            text        : texto completo
            input_ids   : tensor de IDs tokenizados
            token_count : numero real de tokens gerados
    """
    print("=" * 60)
    print("  Passo 2 — Simulando o RAG Massivo")
    print("=" * 60)

    # Ajusta o corpus para atingir o numero alvo de tokens
    text = CORPUS_MEDICO
    while True:
        tokens = tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
        )
        count = tokens["input_ids"].shape[1]
        if count >= target_tokens:
            # Trunca exatamente no alvo
            tokens = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=target_tokens,
            )
            count = tokens["input_ids"].shape[1]
            break
        text = text + "\n\n" + CORPUS_MEDICO

    print(f"\n  Contexto medico gerado:")
    print(f"    Caracteres  : {len(text):,}")
    print(f"    Tokens reais: {count:,}")
    print(f"    Shape tensor: {tokens['input_ids'].shape}")
    print(f"\n  Primeiros 200 caracteres do contexto:")
    print(f"    {text[:200].strip()}...")
    print("=" * 60)

    return {
        "text":        text[:target_tokens * 6],
        "input_ids":   tokens["input_ids"],
        "token_count": count,
    }


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    result = generate_rag_context(tokenizer, target_tokens=12000)
    print(f"\n  Passo 2 OK — Contexto RAG com {result['token_count']:,} tokens pronto")
