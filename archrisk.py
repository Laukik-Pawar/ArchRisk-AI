
"""ArchRisk AI first-evidence prototype. All edges mean consumer -> required provider."""
import argparse
import json
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch import nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
import time
import random
from google.genai.errors import ServerError

ROOT = Path(__file__).resolve().parent
TYPES = ['gateway','service','database','cache','queue','external','other']



def validate(data):
    if not isinstance(data, dict):
        raise ValueError("Architecture must be a JSON object.")

    nodes = data.get("components")
    edges = data.get("dependencies")

    if not isinstance(nodes, list) or len(nodes) < 2:
        raise ValueError("Architecture requires at least two components.")

    if not isinstance(edges, list):
        raise ValueError("Dependencies must be a list.")

    ids = set()

    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError("Each component must be an object.")

        for field in ("id", "name", "type"):
            if not isinstance(node.get(field), str) or not node[field].strip():
                raise ValueError(f"Invalid component field: {field}")

        if node["id"] in ids:
            raise ValueError(f"Duplicate component ID: {node['id']}")

        ids.add(node["id"])

        if node["type"] not in TYPES:
            raise ValueError(f"Unsupported component type: {node['type']}")

        criticality = node.get("criticality", 0.5)

        if (
            isinstance(criticality, bool)
            or not isinstance(criticality, (int, float))
            or not np.isfinite(criticality)
            or not 0 <= criticality <= 1
        ):
            raise ValueError(f"Invalid criticality for {node['id']}")

        node["criticality"] = float(criticality)
        node.setdefault("description", "")

        if not isinstance(node["description"], str):
            raise ValueError(f"Invalid description for {node['id']}")

    for edge in edges:
        if not isinstance(edge, dict):
            raise ValueError("Each dependency must be an object.")

        source = edge.get("source")
        target = edge.get("target")

        if source not in ids or target not in ids:
            raise ValueError(f"Unknown dependency: {source} -> {target}")

        if source == target:
            raise ValueError("Self-dependencies are not supported.")

    return data



def gemini_client():
    """Require Gemini for every architecture analysis; never fall back to reference JSON."""
    if not os.environ.get('GEMINI_API_KEY'):
        raise RuntimeError('GEMINI_API_KEY is required. Set it in your terminal before running.')
    from google import genai
    return genai.Client(api_key=os.environ['GEMINI_API_KEY'])


def call_gemini_with_retry(client, model, contents, config, max_attempts=5):
    for attempt in range(max_attempts):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )

        except ServerError as e:
            if e.code not in (500, 502, 503, 504):
                raise

            if attempt == max_attempts - 1:
                raise RuntimeError(
                    "Gemini is temporarily unavailable after multiple attempts."
                ) from e

            wait = min(2 ** attempt + random.uniform(0, 1), 20)
            print(f"Gemini server busy. Retrying in {wait:.1f} seconds...")
            time.sleep(wait)

def extract_architecture(text):
    """Extract architecture from the supplied text using Gemini only."""
    from google.genai import types
    instruction = """Extract ONLY explicitly described software components and dependencies.
Return a JSON object with keys: system (string), edge_semantics (string),
components (array of objects with id, name, type, description, criticality),
dependencies (array of objects with source, target).
Use unique snake_case component IDs; component type must be one of gateway, service,
database, cache, queue, external, other. Criticality is 0..1 based on explicitly
stated importance; otherwise use 0.5. Edges mean source depends on target.
Do not invent missing dependencies, components, or redundancy. No markdown."""
    #client = gemini_client()

    
    client = gemini_client()

    response = call_gemini_with_retry(
        client=client,
        model=os.environ.get("GEMINI_MODEL", "gemini-3.8-flash"),
        contents=instruction + "\n\nArchitecture description:\n" + text,
        
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0
        ),
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty architecture extraction.")

    data = json.loads(response.text)
    return validate(data)



def build_graph(data):
    g=nx.DiGraph()
    for n in data['components']: g.add_node(n['id'],**n)
    for e in data['dependencies']: g.add_edge(e['source'],e['target'])
    return g


def simulation_scores(g):
    """Simplifying assumption: each required provider failure disables all transitive consumers."""
    return {n:len(nx.ancestors(g,n))/(len(g)-1) if len(g)>1 else 0.0 for n in g.nodes}


def risk_table(g):
    n=max(len(g)-1,1)
    bet=nx.betweenness_centrality(g,normalized=True)
    failure=simulation_scores(g)
    rows=[]
    for node in g:
        indeg=g.in_degree(node)/n; outdeg=g.out_degree(node)/n
        heuristic=.3*indeg+.3*outdeg+.4*bet[node]
        rows.append(dict(id=node,name=g.nodes[node]['name'],type=g.nodes[node].get('type','other'),in_degree=g.in_degree(node),out_degree=g.out_degree(node),betweenness=bet[node],heuristic_score=heuristic,simulation_risk=failure[node],affected_components=len(nx.ancestors(g,node))))
    return pd.DataFrame(rows).sort_values('simulation_risk',ascending=False).reset_index(drop=True)


def make_variants(base, count=120, seed=42):
    """Synthetic graphs retain the same components, varying dependency edges only."""
    rng=np.random.default_rng(seed)
    original=list(base.edges()); nodes=list(base.nodes())
    variants=[]
    for _ in range(count):
        g=base.copy()
        for a,b in original:
            if rng.random()<.18: g.remove_edge(a,b)
        for _ in range(int(rng.integers(0,5))):
            a,b=rng.choice(nodes,2,replace=False)
            if not nx.has_path(g,b,a) and not g.has_edge(a,b): g.add_edge(a,b)
        variants.append(g)
    return variants


def features(g, node_order):
    """No explicit centrality or degree features; structural adjacency is passed separately."""
    vals=[]
    for node in node_order:
        attr=g.nodes[node]; typ=attr.get('type','other')
        onehot=[float(typ==t) for t in TYPES]
        vals.append(onehot+[float(attr.get('criticality',.5)),min(len(attr.get('description',''))/100,1)])
    x=torch.tensor(vals,dtype=torch.float32)
    # Reverse messages so providers can learn from their consumers.
    adj=np.zeros((len(node_order),len(node_order)),dtype=np.float32)
    ix={n:i for i,n in enumerate(node_order)}
    for consumer,provider in g.edges: adj[ix[provider],ix[consumer]]=1
    adj+=np.eye(len(node_order),dtype=np.float32)
    deg=np.sum(adj,axis=1,keepdims=True); adj=adj/np.maximum(deg,1)
    return x,torch.tensor(adj,dtype=torch.float32)


class SimpleGCN(nn.Module):
    """Lightweight message-passing GNN implemented in PyTorch."""
    def __init__(self,dim,hidden=32):
        super().__init__(); self.l1=nn.Linear(dim,hidden);self.l2=nn.Linear(hidden,hidden);self.out=nn.Linear(hidden,1)
    def forward(self,x,a):
        h=torch.relu(self.l1(a@x));h=torch.relu(self.l2(a@h));return torch.sigmoid(self.out(h)).flatten()


def train_and_evaluate(base, variants=120, epochs=160, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    graphs=make_variants(base,variants,seed); split=int(.8*len(graphs))
    train=graphs[:split];test=graphs[split:]
    order=list(base.nodes()); model=SimpleGCN(len(TYPES)+2)
    optimizer=torch.optim.Adam(model.parameters(),lr=.015,weight_decay=1e-4)
    for _ in range(epochs):
        model.train();optimizer.zero_grad();losses=[]
        for g in train:
            x,a=features(g,order);y=torch.tensor([simulation_scores(g)[n] for n in order],dtype=torch.float32)
            losses.append(nn.functional.mse_loss(model(x,a),y))
        loss=torch.stack(losses).mean();loss.backward();optimizer.step()
    model.eval(); preds=[];labels=[];heur=[];indegree_baseline = []
    with torch.no_grad():
        for g in test:
            x,a=features(g,order)
            preds.extend(model(x,a).numpy().tolist())
            scores=simulation_scores(g);labels.extend([scores[n] for n in order])
            ht=risk_table(g).set_index('id')['heuristic_score'];heur.extend([ht[n] for n in order]);denominator=max(len(g)-1,1)
            indegree_baseline.extend([g.in_degree(n)/denominator for n in order])
        x,a=features(base,order);base_preds=model(x,a).numpy()
    corr=spearmanr(preds,labels).statistic
    heuristic_corr=spearmanr(heur,labels).statistic
    
    indegree_corr = spearmanr(
        indegree_baseline, labels
    ).statistic

    indegree_mae = mean_absolute_error(
        labels, indegree_baseline
    )

    metrics={"indegree_baseline_mae": float(indegree_mae),
"indegree_baseline_spearman": (
    float(indegree_corr)
    if np.isfinite(indegree_corr)
    else None
),'train_graphs':len(train),'test_graphs':len(test),'test_nodes':len(labels),'gnn_mae':float(mean_absolute_error(labels,preds)),'gnn_rmse':float(np.sqrt(mean_squared_error(labels,preds))),'gnn_spearman':float(corr) if np.isfinite(corr) else None,'heuristic_spearman':float(heuristic_corr) if np.isfinite(heuristic_corr) else None,'label_definition':'fraction of other nodes transitively dependent on failed component','generalization_limit':'Held-out synthetic variants of ONE architecture, not independent architectures'}
    return model,dict(zip(order,map(float,base_preds))),metrics


def recommendations(g, df):
    """Gemini-only recommendations grounded in graph and simulation results."""
    from google.genai import types
    high=df.sort_values('simulation_risk',ascending=False).head(5)
    evidence=[{'component':r['name'],'simulation_risk':round(float(r['simulation_risk']),3),
               'gnn_predicted_risk':round(float(r['gnn_predicted_risk']),3) if 'gnn_predicted_risk' in df else None,
               'affected':int(r['affected_components']),
               'consumers':[g.nodes[x]['name'] for x in nx.ancestors(g,r['id'])]}
              for _,r in high.iterrows()]
    prompt=('You are a software integration-risk analyst. Provide 4 concise, actionable '
            'recommendations based ONLY on the evidence below. Explain which components '
            'are critical and why. Scores represent simulated impact, NOT failure probability. '
            'Do not invent missing dependencies, redundancy, or availability guarantees. '
            'Evidence: '+json.dumps(evidence))
    
    client = gemini_client()

    response = call_gemini_with_retry(
        client=client,
        model=os.environ.get("GEMINI_MODEL", "gemini-3.8-flash"),
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0),
    )
                                                                                
    if not response.text:
        raise RuntimeError('Gemini returned empty recommendations.')
    return response.text


def draw_graph(g,df,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    risk=df.set_index('id')['simulation_risk']
    fig,ax=plt.subplots(figsize=(13,9));pos=nx.spring_layout(g,seed=7)
    nx.draw_networkx_edges(g,pos,ax=ax,arrows=True,arrowsize=16,alpha=.48,node_size=1650)
    nodes=nx.draw_networkx_nodes(g,pos,ax=ax,node_size=1650,node_color=[risk[n] for n in g],cmap='YlOrRd',vmin=0,vmax=1)
    nx.draw_networkx_labels(g,pos,{n:g.nodes[n]['name'].replace(' Service','\nService').replace(' Database','\nDatabase') for n in g},font_size=7,ax=ax)
    fig.colorbar(nodes,ax=ax,label='Simulated failure impact (fraction of other nodes)',shrink=.65)
    ax.set_title('ArchRisk AI — directed dependencies (consumer → provider)');ax.axis('off');fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


def run(text_file=None, variants=120, epochs=160):
    text=Path(text_file or ROOT/'data'/'architecture.txt').read_text(encoding='utf-8')
    data=extract_architecture(text)
    g=build_graph(data);df=risk_table(g)
    model,pred,metrics=train_and_evaluate(g,variants,epochs)
    df['gnn_predicted_risk']=df['id'].map(pred)
    feedback=recommendations(g,df)
    out=ROOT/'results';out.mkdir(exist_ok=True)
    (out/'extracted_graph.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    df.to_csv(out/'risk_scores.csv',index=False)
    (out/'evaluation.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    (out/'recommendations.md').write_text(feedback,encoding='utf-8')
    draw_graph(g,df,out/'risk_graph.png')
    torch.save(model.state_dict(),out/'gnn_model.pt')
    print('Components:',len(g),'Dependencies:',g.number_of_edges())
    print(df[['name','simulation_risk','heuristic_score','gnn_predicted_risk']].head(10).to_string(index=False))
    print('\nEvaluation:',json.dumps(metrics,indent=2))
    print('Saved outputs to',out)
    return df,metrics

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Gemini-only architecture risk analysis')
    p.add_argument('--text-file');p.add_argument('--variants',type=int,default=120)
    p.add_argument('--epochs',type=int,default=160)
    args=p.parse_args();run(args.text_file,args.variants,args.epochs)
