import streamlit as st
import SSD       # SSD.py 파일
import ensemble1  # ensemble.py 파일

def main():
    st.sidebar.title("실행 옵션")
    choice = st.sidebar.radio("실행할 기능 선택", ("SSD 분석", "Ensemble 실행"))
    
    if choice == "SSD 분석":
        SSD.ssd_run()
    elif choice == "Ensemble 실행":
        ensemble1.ensemble_run()

if __name__ == "__main__":
    main()
