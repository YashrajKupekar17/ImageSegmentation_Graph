"""
Disjoint Set Union (Union-Find) data structure with path compression and union by rank.
Optimized for image segmentation tasks where we need to efficiently merge pixel regions.
"""
# project/core/dsu.py

class DSU:
    def __init__(self, n: int):
        """
        Initialize DSU with n elements.
        
        Args:
            n: Number of elements
        """
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n
    
    def find(self, x: int) -> int:
        """
        Find root of set containing x with path compression.
        Time complexity: O(α(n)) amortized, where α is inverse Ackermann function.
        
        Args:
            x: Element to find root for
            
        Returns:
            Root of the set containing x
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        """
        Union two sets by rank optimization.
        
        Args:
            x: First element
            y: Second element
            
        Returns:
            True if union was performed, False if already in same set
        """
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False
        
        # Union by rank for better tree balance
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
            self.size[root_y] += self.size[root_x]
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
            self.size[root_x] += self.size[root_y]
        else:
            self.parent[root_y] = root_x
            self.size[root_x] += self.size[root_y]
            self.rank[root_x] += 1
        
        return True
    
    def connected(self, x: int, y: int) -> bool:
        """
        Check if two elements are in the same connected component.
        
        Args:
            x: First element
            y: Second element
            
        Returns:
            True if elements are connected
        """
        return self.find(x) == self.find(y)
    
    def get_size(self, x: int) -> int:
        """
        Get size of the connected component containing x.
        
        Args:
            x: Element to get component size for
            
        Returns:
            Size of the component
        """
        return self.size[self.find(x)]
    
    def get_components(self) -> dict:
        """
        Get all unique root representatives and their sizes.
        
        Returns:
            Dictionary mapping root -> size
        """
        components = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in components:
                components[root] = self.size[root]
        return components