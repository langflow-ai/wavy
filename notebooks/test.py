
# dunder_methods = ['__add__', '__sub__', '__mul__', '__ge__', '__gt__', '__le__', '__lt__', '__pow__']

# def _one_arg(list, value, __f):
#     return [getattr(a, __f)(value) for a in list]

# class Test:
#     def __init__(self, blocks):
#         # TODO: blocks must have increasing indexes, add warning and reindex
#         self.list = blocks

#     for dunder in dunder_methods:
#         locals()[dunder] = lambda self, value, __f=dunder, *args, **kwargs: _one_arg(self.list, value, __f)


# test = Test([1,2,3,4])
# print(test.list)

# print(test.__add__(1))
# print(test.__mul__(2))



#     def __getitem__(self, key):
#         start, stop, selection = None, None, None

#         if isinstance(key, int):
#             selection = self.pairs[key]
#             if selection:
#                 return selection

#         elif isinstance(key, str):
#             selection = [
#                 pair
#                 for pair in self.pairs
#                 if pd.Timestamp(pair.xstart) == pd.Timestamp(key)
#             ]
#             if selection:
#                 return selection[0]  # No xstart repeat

#         elif isinstance(key, slice):
#             selection = self.pairs
#             if isinstance(key.start, pd.Timestamp) or isinstance(
#                 key.stop, pd.Timestamp
#             ):
#                 if key.start:
#                     selection = [
#                         pair
#                         for pair in selection
#                         if pd.Timestamp(pair.xstart) >= key.start
#                     ]
#                 if key.stop:
#                     selection = [
#                         pair
#                         for pair in selection
#                         if pd.Timestamp(pair.xstart) < key.stop
#                     ]

#             elif isinstance(key.start, int) or isinstance(key.stop, int):
#                 if key.start:
#                     selection = selection[key.start :]
#                 if key.stop:
#                     selection = selection[: key.stop]

#             elif isinstance(key.start, str) or isinstance(key.stop, str):
#                 if key.start:
#                     selection = [
#                         pair
#                         for pair in selection
#                         if pd.Timestamp(pair.xstart) >= pd.Timestamp(key.start)
#                     ]
#                 if key.stop:
#                     selection = [
#                         pair
#                         for pair in selection
#                         if pd.Timestamp(pair.xstart) < pd.Timestamp(key.stop)
#                     ]

#         if selection:
#             return TimePanel(selection)

def StringChallenge(strParam):

  removed = []

  while len(removed) <= 2:
    if strParam == strParam[::-1]:
      return ''.join(removed)
    elif len(removed) == 2:
        return 'not possible'
    for i in range(int(len(strParam)/2)):
      if strParam[i] != strParam[len(strParam)-i-1]:
        removed.append(strParam[i])
        strParam = strParam[0 : i : ] + strParam[i + 1 : :]
        break

print(StringChallenge('mmop'))