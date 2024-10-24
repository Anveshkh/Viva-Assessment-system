"""
This approach uses Sentence Transformers
"""

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Two lists of sentences
llm_answer = "Einstein's Theory of Relativity fundamentally changed our understanding of space, time, and gravity. It consists of two parts: Special Relativity, introduced in 1905, and General Relativity, developed in 1915. Special Relativity focuses on objects moving at constant speeds, particularly those approaching the speed of light, and introduces the idea that time and space are relative depending on the observer's motion. It also famously established the equation E = mc², demonstrating the equivalence of mass and energy. General Relativity extends these ideas to include gravity, explaining it not as a force but as a curvature of spacetime caused by massive objects like stars and planets. This curvature affects the motion of objects, offering a revolutionary way to describe gravitational effects. Together, these theories form the foundation of modern physics and have been validated by numerous experiments and observations, such as the bending of light by gravity."

user_answer = "Einstein's Theory of Relativity focuses on space, time and gravity"
# Compute embeddings for both lists
embeddings1 = model.encode(llm_answer)
embeddings2 = model.encode(user_answer)

print(embeddings1.shape)
print(embeddings2.shape)
# Compute cosine similarities
similarities = model.similarity(embeddings1, embeddings2)

print(similarities)