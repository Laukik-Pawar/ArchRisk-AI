
"""Streamlit first-evidence dashboard. Run: streamlit run app.py"""
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from archrisk import ROOT, extract_architecture, build_graph, risk_table, simulation_scores, recommendations, train_and_evaluate

st.set_page_config(page_title='ArchRisk AI — First Evidence',layout='wide')
st.title('ArchRisk AI — E-commerce Integration Risk')
st.caption('First-evidence prototype: one reference architecture; mandatory Gemini extraction. Risk estimates are simulation proxies, not failure probabilities.')

text=st.text_area(
    'Architecture description',
    value=(ROOT/'data'/'architecture.txt').read_text(encoding='utf-8'),
    height=170
)

st.info('Every analysis calls Gemini to extract the architecture and generate recommendations. GEMINI_API_KEY is required.')

if st.button('Analyze architecture',type='primary'):
    with st.spinner('Extracting, simulating and training GNN...'):
        try:
            data=extract_architecture(text)
            g=build_graph(data)
            df=risk_table(g)
            model,preds,metrics=train_and_evaluate(g,variants=70,epochs=75)
            df['gnn_predicted_risk']=df['id'].map(preds)
            feedback=recommendations(g,df)

            st.session_state['data']=data
            st.session_state['risk']=df
            st.session_state['metrics']=metrics
            st.session_state['feedback']=feedback

        except Exception as exc:
            st.error(f'Analysis failed: {exc}')

if 'data' in st.session_state:
    data=st.session_state['data']
    g=build_graph(data)
    df=st.session_state['risk']
    metrics=st.session_state['metrics']

    c1,c2,c3=st.columns(3)
    c1.metric('Components',len(g))
    c2.metric('Dependencies',g.number_of_edges())
    c3.metric('Test GNN MAE',f"{metrics['gnn_mae']:.3f}")

    st.subheader('Risk ranking')
    st.dataframe(
        df[['name','type','simulation_risk','heuristic_score',
            'gnn_predicted_risk','affected_components']],
        hide_index=True,
        use_container_width=True
    )

    st.subheader('Architecture graph')
    pos=nx.spring_layout(g,seed=7)
    fig=go.Figure()

    for a,b in g.edges():
        x0,y0=pos[a]
        x1,y1=pos[b]

        fig.add_trace(go.Scatter(
            x=[x0,x1],y=[y0,y1],
            mode='lines',
            line=dict(color='#94a3b8',width=1),
            hoverinfo='skip',
            showlegend=False
        ))

        fig.add_annotation(
            x=x1,y=y1,ax=x0,ay=y0,
            xref='x',yref='y',
            axref='x',ayref='y',
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor='#64748b',
            opacity=.75
        )

    scores=df.set_index('id')['simulation_risk']

    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in g],
        y=[pos[n][1] for n in g],
        mode='markers+text',
        text=[g.nodes[n]['name'] for n in g],
        textposition='top center',
        marker=dict(
            size=22,
            color=[scores[n] for n in g],
            colorscale='YlOrRd',
            cmin=0,cmax=1,
            showscale=True,
            colorbar=dict(title='Impact')
        ),
        hovertext=[
            f"{g.nodes[n]['name']}: {scores[n]:.1%} affected"
            for n in g
        ],
        hoverinfo='text',
        showlegend=False
    ))

    fig.update_layout(
        height=620,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20,r=20,t=30,b=20)
    )

    st.plotly_chart(fig,use_container_width=True)
    st.caption('Arrow direction: consumer → required provider. Failure propagation is evaluated in the reverse direction.')

    st.subheader('Failure simulation')

    selected=st.selectbox(
        'Fail component',
        list(g.nodes),
        format_func=lambda n:g.nodes[n]['name']
    )

    affected=nx.ancestors(g,selected)

    st.warning(
        f"Failure of {g.nodes[selected]['name']} could affect "
        f"{len(affected)} of {len(g)-1} other components "
        f"({simulation_scores(g)[selected]:.1%})."
    )

    st.write(
        ', '.join(sorted(g.nodes[n]['name'] for n in affected))
        or 'No dependent components.'
    )

    st.subheader('Engineering recommendations')
    st.markdown(st.session_state['feedback'])

    st.subheader('Evaluation')
    st.json(metrics)

    st.download_button(
        'Download risk results CSV',
        df.to_csv(index=False),
        file_name='risk_scores.csv',
        mime='text/csv'
    )

    st.download_button(
        'Download graph JSON',
        json.dumps(data,indent=2),
        file_name='extracted_graph.json',
        mime='application/json'
    )
