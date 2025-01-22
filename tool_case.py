# #################### КЛАСС Tool #########################
class Tool:
    """
    Класс - обертка для работы с моделями и другими инструментами
    """
    tools_name_list = []     # список имен инструментов в классе

    @staticmethod
    def display_tool_count():
        print("Экземпляров класса Tool (загруженных инструментов), ВСЕГО: {}".format(len(Tool.tools_name_list)))

    @staticmethod
    def reset_tool_count():
        tools_count_prev = len(Tool.tools_name_list)
        Tool.tools_name_list = []
        print("Счетчик экземпляров класса Tool (список имен) обнулен: {} -> {}".format(tools_count_prev,
                                                                                       len(Tool.tools_name_list)))

    def __init__(self, tool_name, tool_obj, tool_type='default'):
        #
        if tool_name not in Tool.tools_name_list:
            self.name = tool_name       # имя инструмента для удобного обращения
            self.tool = tool_obj        # инструмент (объект)
            #
            self.type = tool_type  # тип инструмента
            if self.type == 'model':
                self.model = self.tool  # тип инструмента модель можно вызывать model
            #
            self.counter = 0            # счетчик вызова инструмента
            #
            Tool.tools_name_list.append(tool_name)
            #
            print("ИНСТРУМЕНТ {} сохранен в классе Tool".format(self.name))
        else:
            raise Exception("Запрещено создавать экземпляр класса Tool с существующим именем")

    def __del__(self):
        if hasattr(self, 'name'):
            if self.type == 'model':

                print("Удалена МОДЕЛЬ (экз.класса Tool): {}".format(self.name))
            else:
                print("Удален ИНСТРУМЕНТ (экз.класса Tool): {}".format(self.name))
        else:
            print("У удаляемого экземпляра класса Tool нет атрибута name")

    def display_tool_info(self):
        if self.type == 'model':
            print("МОДЕЛЬ: {}, количество предиктов {}".format(self.name, self.counter))
        else:
            print("ИНСТРУМЕНТ: {}, количество обращений {}".format(self.name, self.counter))


def get_tool_by_name(tool_name, tool_list):
    """
    Ищет инструмент по имени среди инструментов в списке экземпляров класса Tool
    Если tool по имени не найден, то возвращает None
    :param tool_name: имя инструмента
    :param tool_list: список экземпляров класса
    :return: модель (экземпляр класса) или None
    """
    for tool in tool_list:
        if tool.tool is not None:
            if tool.name == tool_name:
                return tool
    return None
