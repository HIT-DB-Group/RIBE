from settings import Settings
from grammar_utils import GrammarParser, TextGenerator, TreeToText ,grammar_complement
from settings import PostgreSQL

def random_text_generate(database):
    grammar_complement(database=database)
    grammar = GrammarParser()
    generator = TextGenerator(grammar,database)
    transformer = TreeToText(database=database)
    count = 0
    texts = []
    print(f"Generate {Settings.text_count} sql ..." )
    while count < Settings.text_count:
        sequence, _ = generator.generate()
        if sequence:
            if count%100==0:
                print(count)
            count += 1
            text = grammar.construct_text(sequence)
            text = (" ".join(text)) 
            tree = grammar.parser.parse(text)
            text = transformer.transform(tree)
            text = (" ".join(text))
            text = text.replace(" . ",".")
            texts.append(text)
    return texts


def test():
    texts = random_text_generate(database= Settings.imdb)
    postgre = PostgreSQL(Settings.database_imdb['database_url'])

    with open("rand.sql","w") as f :
        f.write(";\n".join(texts)) 
    for text in texts:
        card,cost = postgre.parser(text)
        print("card:",card)
        print("cost:",cost)
test()