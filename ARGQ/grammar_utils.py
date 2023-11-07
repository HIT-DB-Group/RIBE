from lark import Lark, Token, Tree, Transformer
from settings import Settings
import torch
from typing import Dict
from shutil import copyfile

from settings import Settings,DataBase



from settings import Settings


def grammar_complement(database):
    copyfile(Settings.basic_sql_lark_path, Settings.convert_lark_path)
    with Settings.convert_lark_path.open("a") as f:
        # Fill table rule
        f.write("tablename:")
        f.write(
            "|".join([f"\"{table}\"" for table in database.table_name]))
        f.write("\n")
        # Fill value rule
        f.write("value:")
        f.write("|".join(["\"{}\"".format(str(i))
                for i in range(Settings.value_sample_num)]))
        f.write("\n")
        # Fill column rule
        f.write("columnname:")
        f.write("|".join([f"\"{column}\"" for column in database.column_name]))
        f.write("\n")
        f.write("joincolumnname:")
        f.write("|".join([f"\"{column}\"" for column in database.join_column]))


class SimpleQuery:
    def __init__(self, id):
        self.id = id
        self.tables = []
        self.field_count = 0
        self.projection = []
        self.aggregation = []
        # Flag
        self.flag_subquery = False
        self.flag_tableref = False
        self.flag_selfield = False
        self.flag_project = False
        self.flag_aggregate = False
        self.flag_joinfield = False
        self.flag_group = False
        self.flag_expression = False
        self.flag_having = False

    def get_column_to_choose(self, database):
        column_to_choose = dict()
        field_count = 0
        for table in self.tables:
            t_num = len(database.table_column[table])
            if t_num > 0:
                column_to_choose[table] = database.table_column[table][:]
                field_count += len(database.table_column[table])
        return column_to_choose, field_count

    def get_group(self):
        return len(self.projection), self.projection[:]


class GrammarParser:
    def __init__(self, grammar=Settings.convert_lark_path.open("r").read()):
        self.parser = Lark(grammar, keep_all_tokens=True)
        self.rules = self.parser.rules
        self.rules_list = []
        self.get_rule_mask()
        # print(self.mask)
        for rule in self.rules:
            rule_item = [rule.origin.name.value.lower()]
            for symbol in rule.expansion:
                if symbol.is_term == True:
                    rule_item.append(self.parser.get_terminal(
                        symbol.name).pattern.value.lower())
                else:
                    rule_item.append(symbol.name.lower())
            self.rules_list.append(rule_item)

    def construct_text(self, rules):
        text = []
        start = self.rules[rules[0]].origin
        symbol_stack = [start]
        rules_stack = list(reversed(rules))
        while symbol_stack:
            symbol = symbol_stack.pop()
            if symbol.is_term == True:
                text.append(self.parser.get_terminal(
                    symbol.name).pattern.value)
                continue
            rule = self.rules[rules_stack.pop()]
            for child in reversed(rule.expansion):
                symbol_stack.append(child)
        return text

    def convert_to_origin_text(self, rules):
        text = []
        start = self.rules[rules[0]].origin
        symbol_stack = [start]
        rules_stack = list(reversed(rules))
        while symbol_stack:
            symbol = symbol_stack.pop()
            if symbol.is_term == True:
                text.append(self.parser.get_terminal(
                    symbol.name).pattern.value)
                continue
            rule = self.rules[rules_stack.pop()]
            for child in reversed(rule.expansion):
                symbol_stack.append(child)
        return text

    def get_rules_sequences(self, tree: Tree):
        stack = [tree]
        parent_stack = [Settings.PADDING_ACTION]
        rules_sequence = []
        parent_rules_sequence = []
        while stack:
            node = stack.pop()
            rule = self.get_rule_from_tree(node)
            rules_sequence.append(rule)
            parent_rules_sequence.append(parent_stack.pop())
            for child in reversed(node.children):
                if isinstance(child, Tree):
                    stack.append(child)
                    parent_stack.append(rule)
        return rules_sequence, parent_rules_sequence

    def get_rule_from_tree(self, tree):
        head = tree.data.value
        rule = [head]
        for child in tree.children:
            if isinstance(child, Tree):
                rule.append(child.data.value)
            elif isinstance(child, Token):
                rule.append(child.value)
            else:
                pass
        return rule

    def rules_to_index(self, rules):
        index_sequence = []
        for rule in rules:
            index = self.rules_list.index(rule)
            index_sequence.append(index)
        return index_sequence

    def get_rule_mask(self):
        self.mask = dict()
        self.rules_count = len(self.rules)
        for index, rule in enumerate(self.rules):
            head = rule.origin
            if head not in self.mask.keys():
                mask_item = torch.ones(self.rules_count).bool()
                self.mask[head] = mask_item
            self.mask[head][index] = False

    def mask_single_rule(self, rule_to_mask):
        mask = torch.zeros(self.rules_count).bool()
        for index, rule in enumerate(self.rules_list):
            if rule_to_mask == rule:
                mask[index] = True
                return mask

    def mask_multiple_rules(self, head, body_list, single=True):
        mask = torch.zeros(self.rules_count).bool()
        for body in body_list:
            if single:
                rule_to_mask = [head]+[body]
            else:
                rule_to_mask = [head] + body
            for index, rule in enumerate(self.rules_list):
                if rule == rule_to_mask:
                    mask[index] = True
                    break
        return mask

    def get_text_rule_sequence(self, text):
        return Settings.Rule(text, text)


class TextGenerator:
    def __init__(self, grammar: GrammarParser,database:DataBase):
        self.grammar = grammar
        self.database = database
        self.rules = grammar.rules
        self.mask = grammar.mask
        self.rules_count = len(self.rules)
        self.start = self.rules[0].origin

    def new(self):
        self.symbol_stack = [Settings.Rule(
            self.start, Settings.PADDING_ACTION)]
        self.semantic_mask = torch.zeros(self.rules_count).bool()
        self.check = Semantic(grammar=self.grammar,database=self.database)
        self.rule_chosen = None

    def generate(self):
        self.new()
        rules = []
        parents = []
        prob = torch.rand(self.rules_count)
        rule, parent = self.step(prob)
        while rule is not None:
            rules.append(rule)
            parents.append(parent)
            rule, parent = self.step(prob)
            if len(rules) > Settings.max_squence_length:
                return [], []
        return rules, parents

    def step(self, prob):
        if self.symbol_stack:
            symbol, parent = self.symbol_stack.pop()
            syntax_mask = self.mask[symbol]
            unit_mask = torch.logical_or(syntax_mask, self.semantic_mask)
            prob = prob.masked_fill(mask=unit_mask, value=0.0)
            rule = torch.multinomial(prob, 1).item()
            self.rule_chosen = self.rules[rule]
            # print(self.rule_chosen)
            for child in reversed(self.rule_chosen.expansion):
                if child.is_term == False:
                    self.symbol_stack.append(Settings.Rule(child, rule))
            self.semantic_mask = self.check.step(self)
            return rule, parent
        else:
            return None, None


class Semantic:
    simple_queris: Dict[int, SimpleQuery]

    def __init__(self, grammar: GrammarParser, database:DataBase):
        self.database = database
        self.grammar = grammar
        self.level = -1
        self.simple_queris = dict()
        self.all_tables = set(self.database.table_name[:])
        self.all_columns = set(self.database.column_name[:])
        self.join_columns = set(self.database.join_column[:])
        self.flag_subquery = False

    def __default__(self, generator: TextGenerator):
        return torch.zeros(generator.rules_count)

    def step(self, generator: TextGenerator):
        #print(generator.rule_chosen)
        return getattr(self, generator.rule_chosen.origin.name.value, self.__default__)(generator)

    def selectstmtfromtable(self, generator: TextGenerator):
        self.level += 1
        self.query = SimpleQuery(id=self.level)
        self.simple_queris[self.level] = self.query
        self.query.flag_tableref = True
        self.table_count = len(self.all_tables)
        if self.table_count == 1:
            return self.grammar.mask_single_rule(['tablerefs', 'tablerefs', ',', 'tablename'])
        else:
            return torch.zeros(generator.rules_count)

    def selectstmtbasic(self, generator: TextGenerator):
        self.query.flag_tableref = False
        self.query.flag_selfield = True
        ''' if self.flag_subquery:
            return self.grammar.mask_single_rule(["selectstmtfieldlist", "*"]) '''
        return torch.zeros(generator.rules_count)

    def tablerefs(self, generator: TextGenerator):
        # tablerefs : tablerefs "," tablename | tablename
        first_symbol = generator.rule_chosen.expansion[0].name.lower()
        self.table_count -= 1
        if first_symbol == "tablerefs":
            # tablerefs : tablerefs "," tablename
            if self.table_count == 1:
                return self.grammar.mask_single_rule(['tablerefs', 'tablerefs', ',', 'tablename'])
        elif first_symbol == "tablename":
            # tablerefs : tablename
            if self.flag_subquery and self.subtable:
                table_to_choose = set([self.subtable])
                table_to_mask = self.all_tables-table_to_choose
                return self.grammar.mask_multiple_rules(head="tablename", body_list=table_to_mask)
        return torch.zeros(generator.rules_count)

    def queryend(self, generator: TextGenerator):
        self.level -= 1
        self.flag_subquery = False
        self.query.flag_group = False
        if self.level != -1:
            self.query = self.simple_queris[self.level]
        return torch.zeros(generator.rules_count)

    def havingclause(self, generator: TextGenerator):
        if generator.rule_chosen.expansion:
            self.query.flag_having = True
            self.expression_count = 0
        return torch.zeros(generator.rules_count)

    def havingend(self, generator: TextGenerator):
        if (self.query.projection and self.query.aggregation) or (self.query.flag_having and self.query.projection):
            self.query.flag_group = True
            self.group_size, self.group_field = self.query.get_group()
            return self.grammar.mask_single_rule(["selectstmtgroup"])
        self.query.flag_having = False
        return self.grammar.mask_single_rule(["selectstmtgroup", "group", "by", "bylist"])

    def selectstmtfieldlist(self, generator: TextGenerator):
        self.column_to_choose, self.field_count = self.query.get_column_to_choose(
            self.database)
        if generator.rule_chosen.expansion[0].name.lower() == "selectfieldlist":
            if self.field_count == 1 or self.flag_subquery:
                return self.grammar.mask_single_rule(["selectfieldlist", "selectfieldlist", ",", "selectstmtfield"])
        else:
            for table, columns in self.column_to_choose.items():
                for column in columns:
                    self.query.projection.append((table, column))
        return torch.zeros(generator.rules_count)

    def selectfieldlist(self, generator: TextGenerator):
        self.field_count -= 1
        # Rule selectfieldlist : selectfieldlist "," selectstmtfield | selectstmtfield
        if self.field_count == 1:
            return self.grammar.mask_single_rule(["selectfieldlist", "selectfieldlist", ",", "selectstmtfield"])
        return torch.zeros(generator.rules_count)

    def selectstmtfield(self, generator: TextGenerator):
        if generator.rule_chosen.expansion[0].name == "tablecolumn":
            self.query.flag_project = True
        else:
            self.query.flag_aggregate = True
            body_list = [["count", "(", 'tablecolumn', ")"], [
                "count", "(", "*", ")"]]
            return self.grammar.mask_multiple_rules("function", body_list, False)
        return torch.zeros(generator.rules_count)

    def function(self, generator: TextGenerator):
        self.query.flag_aggregate = True
        return torch.zeros(generator.rules_count)

    def selectend(self, generator: TextGenerator):
        self.query.flag_selfield = False
        self.join_size = len(self.query.tables)-1
        if self.join_size > 0:
            self.query.flag_joinfield = True
            return self.grammar.mask_single_rule(["whereclauseoptional"])
        return torch.zeros(generator.rules_count)

    def whereclauseoptional(self, generator: TextGenerator):
        if self.join_size > 0:
            self.join_tables = self.query.tables[:]
            self.joined_tables = []
            rule_to_mask = ['whereclause', 'expression']
        elif self.join_size == 0:
            rule_to_mask = ['whereclause', 'joinfield', 'logand', 'expression']
        return self.grammar.mask_single_rule(rule_to_mask)

    def whereclause(self, generator: TextGenerator):
        # whereclause : [joinfield logand] expression
        self.expression_count = 0
        if self.join_size == 1:
            return self.grammar.mask_single_rule(['joinfield', 'joinfield', 'logand', 'join'])
        elif self.join_size > 1:
            return self.grammar.mask_single_rule(['joinfield', 'join'])
        return torch.zeros(generator.rules_count)

    def whereend(self, generator: TextGenerator):
        self.query.flag_expression = False
        return torch.zeros(generator.rules_count)

    ''' def orderbyoptional(self, generator: TextGenerator):
        self.flag_orderby = True
        self.ordersize = Settings.max_orderby_num
        return torch.zeros(generator.rules_count) '''

    def selectstmtgroup(self, generator: TextGenerator):
        if self.query.flag_group:
            if self.group_size == 1:
                return self.grammar.mask_single_rule(["bylist", "bylist", ",", "byitem"])
            elif self.group_size > 1:
                return self.grammar.mask_single_rule(["bylist",  "byitem"])
        return torch.zeros(generator.rules_count)

    def bylist(self, generator: TextGenerator):
        if self.query.flag_group:
            self.group_size -= 1
            if self.group_size == 1:
                return self.grammar.mask_single_rule(["bylist", "bylist", ",", "byitem"])
            else:
                return self.grammar.mask_single_rule(["bylist",  "byitem"])
        ''' elif self.flag_orderby:
            self.ordersize -= 1
            if self.ordersize == 1:
                return self.grammar.mask_single_rule(["bylist", "bylist", ",", "byitem"]) '''
        return torch.zeros(generator.rules_count)

    def inornotop(self, generator: TextGenerator):
        self.flag_subquery = True
        self.subtable = self.table_chosen
        self.subcolumn = self.column_chosen
        return torch.zeros(generator.rules_count)

    def anyorall(self, generator: TextGenerator):
        self.flag_subquery = True
        self.subtable = self.table_chosen
        self.subcolumn = self.column_chosen
        return torch.zeros(generator.rules_count)

    def joinfield(self, generator: TextGenerator):
        self.join_size -= 1
        if self.join_size == 1:
            return self.grammar.mask_single_rule(['joinfield', 'joinfield', 'logand', 'join'])
        return torch.zeros(generator.rules_count)

    def join(self, generator: TextGenerator):
        self.join_state = 0
        return torch.zeros(generator.rules_count)

    def expression(self, generator: TextGenerator):
        self.query.flag_joinfield = False
        self.query.flag_expression = True
        self.expression_count += 1
        if self.expression_count == Settings.max_expression_num-1:
            return self.grammar.mask_single_rule(['expression', 'expression', 'logand', 'boolpri'])
        return torch.zeros(generator.rules_count)

    def haveexpresion(self, generator: TextGenerator):
        self.expression_count += 1
        if self.expression_count == Settings.max_expression_num-1:
            return self.grammar.mask_single_rule(['haveexpression', 'haveexpression', 'logand', 'haveboolpri'])
        return torch.zeros(generator.rules_count)

    def joinop(self, generator: TextGenerator):
        self.join_state = 1
        return torch.zeros(generator.rules_count)

    def tablecolumn(self, generator: TextGenerator):
        if self.query.flag_selfield:
            if self.flag_subquery:
                table_to_choose = set([self.subtable])
            else:
                table_to_choose = set([])
                for table, columns in self.column_to_choose.items():
                    if columns:
                        table_to_choose.add(table)
            table_to_mask = self.all_tables-table_to_choose
            return self.grammar.mask_multiple_rules('tablename', table_to_mask)

        elif self.query.flag_expression:
            table_to_choose = set(self.query.tables)
            table_to_mask = self.all_tables-table_to_choose
            return self.grammar.mask_multiple_rules('tablename', table_to_mask)
        elif self.query.flag_group:
            table_to_choose = set([self.group_field[-1][0]])
            table_to_mask = self.all_tables-table_to_choose
            return self.grammar.mask_multiple_rules('tablename', table_to_mask)

        elif self.query.flag_having:
            table_to_choose = set(self.query.tables)
            table_to_mask = self.all_tables-table_to_choose
            return self.grammar.mask_multiple_rules('tablename', table_to_mask)
        ''' elif self.flag_orderby:
            table_to_choose = set(self.query.tables)
            table_to_mask = self.all_tables-table_to_choose
            return self.grammar.mask_multiple_rules('tablename', table_to_mask) '''
        return torch.zeros(generator.rules_count)

    def jointablecolumn(self, generator: TextGenerator):
        if self.join_state == 0:

            table_to_choose = set()
            if self.joined_tables:
                for t in self.join_tables:  
                    for table in self.joined_tables:  
                        if table in self.database.join_key[t].keys():
                            table_to_choose.add(table)
                            continue
            else:
                table_to_choose = set(self.join_tables)
        elif self.join_state == 1:
            table_to_choose = set()
            for table, joinkey in self.database.join_key[self.table_chosen].items():
                if table in self.join_tables:  
                    for key in joinkey:
                        if key[0] == self.column_chosen:  
                            table_to_choose.add(table)
        table_to_mask = self.all_tables-table_to_choose
        return self.grammar.mask_multiple_rules('tablename', table_to_mask)

    def tablename(self, generator: TextGenerator):
        self.table_chosen = generator.rule_chosen.expansion[0].name.lower()
        if self.query.flag_tableref:
            self.query.tables.append(self.table_chosen)
            chosen_tables = set(self.query.tables)
            have_join_tables = set()
            for table in chosen_tables:
                for t in self.database.join_key[table].keys():
                    have_join_tables.add(t)
            table_to_mask = (self.all_tables-have_join_tables) | chosen_tables
            return self.grammar.mask_multiple_rules("tablename", table_to_mask)

        elif self.query.flag_selfield:
            if self.flag_subquery and self.subcolumn:
                column_to_choose = set([self.subcolumn])
            else:
                column_to_choose = set(
                    self.column_to_choose[self.table_chosen][:])
            column_to_mask = self.all_columns - column_to_choose
            return self.grammar.mask_multiple_rules("columnname", column_to_mask)
        elif self.query.flag_joinfield:
            if self.join_state == 0:
                if self.table_chosen not in self.joined_tables:
                    self.joined_tables.append(self.table_chosen)
                    self.join_tables.remove(self.table_chosen)
                column_to_choose = set()
                for table, joinkey in self.database.join_key[self.table_chosen].items():
                    if table in self.join_tables:
                        for key in joinkey:
                            column_to_choose.add(key[0])
            elif self.join_state == 1:
                self.joined_tables.append(self.table_chosen)
                self.join_tables.remove(self.table_chosen)
                column_to_choose = set()
                for key in self.database.join_key[self.old_table_chosen][self.table_chosen]:
                    column_to_choose.add(key[1])
            column_to_mask = self.join_columns - column_to_choose
            return self.grammar.mask_multiple_rules("joincolumnname", column_to_mask)
        elif self.query.flag_expression:
            column_to_choose = set(
                self.database.table_column[self.table_chosen])
            column_to_mask = self.all_columns-column_to_choose
            return self.grammar.mask_multiple_rules("columnname", column_to_mask)
        elif self.query.flag_group:
            column_to_choose = set([self.group_field[-1][1]])
            column_to_mask = self.all_columns-column_to_choose
            self.group_field.pop()
            return self.grammar.mask_multiple_rules("columnname", column_to_mask)
        elif self.query.flag_having:
            column_to_choose = set(
                self.database.table_column[self.table_chosen])
            column_to_mask = self.all_columns-column_to_choose
            return self.grammar.mask_multiple_rules("columnname", column_to_mask)
        ''' elif self.flag_orderby:
            column_to_choose = set(
                self.database.table_column[self.table_chosen])
            column_to_mask = self.all_columns-column_to_choose
            return self.grammar.mask_multiple_rules("columnname", column_to_mask) '''
        return torch.zeros(generator.rules_count)

    def funcop(self, generator: TextGenerator):
        if self.query.flag_having:
            if self.database.tables[self.table_chosen].columns[self.column_chosen].dtype == "string":
                return self.grammar.mask_multiple_rules(head='compareop', body_list=["<=", "<",  ">=",  ">"])
        return torch.zeros(generator.rules_count)

    def columnname(self, generator: TextGenerator):
        self.old_table_chosen = self.table_chosen
        self.column_chosen = generator.rule_chosen.expansion[0].name.lower()
        if self.query.flag_selfield:
            self.column_to_choose[self.table_chosen].remove(self.column_chosen)
            if self.query.flag_aggregate:
                self.query.flag_aggregate = False
                self.query.aggregation.append(
                    (self.table_chosen, self.column_chosen))
                if self.database.tables[self.table_chosen].columns[self.column_chosen].dtype == "string":
                    return self.grammar.mask_multiple_rules(head='funcop', body_list=["AVG", "SUM"])
            elif self.query.flag_project:
                self.query.flag_project = False
                self.query.projection.append(
                    (self.table_chosen, self.column_chosen))

        elif self.query.flag_having:
            if self.query.flag_aggregate:
                self.query.flag_aggregate = False
                self.query.aggregation.append(
                    (self.table_chosen, self.column_chosen))
                if self.database.tables[self.table_chosen].columns[self.column_chosen].dtype == "string":
                    return self.grammar.mask_multiple_rules(head='funcop', body_list=["AVG", "SUM"])
            else:
                self.query.projection.append(
                    (self.table_chosen, self.column_chosen))

        elif self.query.flag_expression:
            if self.database.tables[self.table_chosen].columns[self.column_chosen].dtype == "string":
                return self.grammar.mask_multiple_rules(head='compareop', body_list=["<=", "<",  ">=",  ">"])

        return torch.zeros(generator.rules_count)

    def joincolumnname(self, generator: TextGenerator):
        self.old_table_chosen = self.table_chosen
        self.column_chosen = generator.rule_chosen.expansion[0].name.lower()
        return torch.zeros(generator.rules_count)
        

class TreeToText(Transformer):
    def __init__(self,database):
        super().__init__()
        self.database = database

    def __default_token__(self, token):
        return token.value

    def function(self, children: list):
        children.reverse()
        children.insert(1, "(")
        children.append(")")
        return self.__default__(None, children, None)

    def tablename(self, children):
        self.table = children[0].lower()
        return self.__default__(None, children, None)

    def columnname(self, children):
        self.column = children[0].lower()
        return self.__default__(None, children, None)

    def value(self, children):
        index = int(children[0])
        # val = self.database.tables[self.table].columns[self.column].value[index]
        try:
            val = self.database.tables[self.table].columns[self.column].value[index]
        except:
            val = index
        children[0] = str(val)
        return self.__default__(None, children, None)

    def selectstmtfromtable(self, children):
        text = []
        selectstmtbasic = children.pop(2)
        children.insert(0, selectstmtbasic)
        havingclause = children.pop(-4)
        children.insert(-1, havingclause)
        for child in children:
            if isinstance(child, list):
                text += child
            elif child is not None:
                text.append(child)
        return text

    def __default__(self, data, children, meta):
        text = []
        for child in children:
            if isinstance(child, list):
                text += child
            elif child is not None:
                text.append(child)
        return text
