import ast
import pandas as pd
import networkx as nx

# Clean the knowledge set
def categoryClean(df , targetCategory):
        
        targetCategoryLen = len(targetCategory)

        # Count the "Category" value containing "targetCategory"
        comment_count = df[df['Category']== targetCategory].shape[0]

        filtered_data = df[df['Category'].str.contains(targetCategory) & (df['Category'].str.len() > targetCategoryLen)]

        targetRow_ids = df[df['Category'].str.contains(targetCategory) & (df['Category'].str.len() > targetCategoryLen)].index

        distinct_categories = df[df['Category'].str.contains(targetCategory)]['Category'].unique()
        
        ## update 
        df.loc[df['Category'].str.contains(targetCategory), 'Entity'] = df['Entity'] + df['Category'].str.split(targetCategory).str[0]
        df['Category'] = df['Category'].apply(lambda x: targetCategory if targetCategory in x else x)
        
        # Count the "Category" value containing "targetCategory"
        comment_count = df[df['Category'] == targetCategory].shape[0]

        filtered_data = df[df['Category'].str.contains(targetCategory) & (df['Category'].str.len() > targetCategoryLen)]

        distinct_categories = df[df['Category'].str.contains(targetCategory)]['Category'].unique()

        return df
    
def load_knowledge_set(path):
    data = []
    # Open the file and parse each line from string to tuple
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():  # Ensure the line is not empty
                try:
                    # Convert string representation of tuple to actual tuple
                    tuple_data = ast.literal_eval(line.strip())
                    data.append(tuple_data)
                except SyntaxError:
                    print(f"Skipping malformed line: {line.strip()}")

    # Load the data into a DataFrame
    df = pd.DataFrame(data, columns=['Entity', 'Category', 'Answer'])
    
    # Clean category
    cateList = ['评论', '主演','口碑','导演','类型','评分','国家地区','适合吃','特色菜','人均价格', '订单量','地址',]
    for cat in cateList:
        df = categoryClean(df, cat)
    
    # Return knowledge set dataframe
    return df

# Construct & Return Knowledge Tree  
def build_tree(df):
    G = nx.DiGraph()
    
    # Add nodes and edges based on the DataFrame
    for index, row in df.iterrows():
        entity_id = f"Entity: {row['Entity']}"
        category_id = f"Category: {row['Category']} ({row['Entity']})"
        answer_id = f"Answer: {row['Answer']} ({row['Category']})"

        # Ensure that nodes for each level (entity, category) are unique per entity-category pair
        if entity_id not in G:
            G.add_node(entity_id, type='Entity', name=row['Entity'])
        if category_id not in G:
            G.add_node(category_id, type='Category', name=row['Category'])
    
        # Answers can be multiple per category, so they are always added
        G.add_node(answer_id, type='Answer', content=row['Answer'])
    
        # Connect nodes hierarchically
        G.add_edge(entity_id, category_id)
        G.add_edge(category_id, answer_id)
    
    # Return Tree
    return G