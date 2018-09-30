# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 08:53:28 2018

@author: mudit
"""

class Node:

    def __init__(self, data, nextNode=None):
        self.data = data
        self.nextNode = nextNode

    def getData(self):
        return self.data

    def setData(self, val):
        self.data = val

    def getNextNode(self):
        return self.nextNode

    def setNextNode(self, val):
        self.nextNode = val


class LinkedList:

    def __init__(self, head=None):
        self.head = head
        self.size = 0

    def getSize(self):
        return self.size

    def printNode(self):
        curr = self.head
        while curr:
            print("\t" + str(curr.data))
            curr = curr.getNextNode()

    def addNode_start(self, data):
        newNode = Node(data, self.head)
        self.head = newNode
        self.size += 1
        print("\tAdded " + str(data))
        return True

    def addNode_end(self, data):
        newNode = Node(data)
        if self.head is None:
            self.head = newNode
            return
        last = self.head
        while (last.getNextNode()):
            last = last.getNextNode()
        last.setNextNode(newNode)
        self.size += 1
        print("\tAdded " + str(data))
        return True

    def findNode(self, value):
        curr = self.head
        while curr:
            if curr.getData() == value:
                return True
            curr = curr.getNextNode()
        return False

    def removeNode(self, value):
        prev = None
        curr = self.head
        while curr:
            if curr.getData() == value:
                if prev:
                    prev.setNextNode(curr.getNextNode())
                else:
                    self.head = curr.getNextNode()
                self.size -= 1
                print("\t\tRemoved" + str(value))
                return True
            prev = curr
            curr = curr.getNextNode()
        return False


myList = LinkedList()
## inserting nodes
print("Inserting")
myList.addNode_start(5)
myList.addNode_start(15)
myList.addNode_end(25)
myList.addNode_end(42)

## printing nodes
print("\n\nPrinting list order")
myList.printNode()

## removing nodes
print("\n\nRemoving")
myList.removeNode(25)

## printing nodes
print("\n\nPrinting list order")
myList.printNode()


## removing nodes
print("\n\nRemoving")
myList.removeNode(15)
myList.removeNode(5)
myList.removeNode(42)

## printing nodes
print("\n\nPrinting list order")
myList.printNode()

print("\n\nSize")
print("\t" + str(myList.getSize()))