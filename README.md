# HealthSearch — Motor de Busca Híbrido

Projeto acadêmico baseado no Desafio Integrador HealthSearch da UNIPÊ.

## Requisitos

- Python 3.10 ou 3.11
- VS Code
- Internet na primeira execução para baixar o modelo de embeddings

## Instalação no VS Code

Abra a pasta do projeto no VS Code e execute no terminal:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Se o PowerShell bloquear a ativação do ambiente, use:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Depois:

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
```

## Executar

```powershell
streamlit run healthsearch_app.py
```

O Streamlit abrirá o endereço local no navegador, normalmente:

http://localhost:8501

## Estrutura

- `healthsearch_app.py` — aplicação completa
- `requirements.txt` — dependências
- `README.md` — instruções
