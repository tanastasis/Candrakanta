import streamlit as st
from .T1 import lx, am, give_paradigm
import traceback
import pandas as pd

st.set_page_config(layout='wide')


# --- Streamlit UI ---
st.title("Paradigm viewer")
st.markdown(
    "Введите lexeme в поле ниже. Приложение вызовет give_paradigm(lexeme) и отобразит результат в виде таблицы.\n\n"
    "**Важно**: убедитесь, что в этом файле определены (или импортированы) переменные/функции: `lx`, `am`, `Grammeme`, `buildForm`, `phonol`, `show_form`."
)

lex = st.text_input("Lexeme", value="", placeholder="введите ключ из lx, например 'lemma1'")
run = st.button("Построить парадигму")

if run and lex.strip():
    try:
        data = give_paradigm(lex.strip())
    except Exception as e:
        st.error("Ошибка при вычислении парадигмы: \n" + traceback.format_exc())
    else:
        # Normalize and display data depending on shape
        try:
            if isinstance(data, list) and len(data) > 0 and all(isinstance(r, list) for r in data):
                # if first row looks like header (first row length >1 and contains strings)
                header = data[0]
                body = data[1:]
                # ensure rows have same length as header (pad with empty strings)
                norm_rows = [r + [""]*(len(header)-len(r)) if len(r) < len(header) else r[:len(header)] for r in body]
                df = pd.DataFrame(norm_rows, columns=header)
                st.dataframe(df, use_container_width=True)
            else:
                # For simple lists or single strings
                st.write("Result:")
                st.write(data)
        except Exception:
            st.write("Не удалось корректно отобразить результат. Вот сырой вывод:")
            st.write(data)