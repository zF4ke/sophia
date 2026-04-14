# dump of live test it should be able to do
might need to have new tools to get these information (like do math tool to do reliable calculations) or might need to enrich the current tools to give all information. or add new finer grained tools to get specific information - i want almost all information to be present but we can add them as we need.

---

Contar caracteres de nomes  (pelo menos na tentativa específica);,
Acessar o conteúdo de canais específicos ,
Somar canais de categorias diferentes ,
Responder sem redundância ,
Encontrar mensagens pelo ID ,
Enxergar cargos   (faltou o asterisco),
Ver a cor do cargo ,
Acesso às datas 

---

# O Problema dos 4 Serviços
O teste consiste em pergunta à Sophia por um resumo dos canais que estão no Serviços (sem referir que é uma categoria), pedindo também por uma descrição de cada um.

### Objetivos:
- Testar planeamento e composição de ferramentas. 
- Testar a ferramenta de listar canais e categorias do servidor.
- Testar a ferramenta de procurar canais e categorias do servidor com base no que foi perguntado.
- Testar a ferramenta de procurar mensagens em canais.
- Testar limites no Runtime: número máximo de chamadas de ferramenta, tempo de resposta, etc.

### Prompt:
```
@Sophia estou com pressa e precisava de um resumo do que tem nos canais do Serviços. quero uma descrição de cada serviço
```

### Resposta esperada (aproximadamente):
```
Oi, F4zke! Entendo a pressa, aqui vai um resumo rápido do que temos na categoria de Serviços:

#2to0: É um servidor de quebra-cabeças com uma pegada psicodélica, ambientado em uma casa paradoxal. Ele é o sucessor do 1to0, criado pelo One Person, e atualmente está ativo em desenvolvimento.
#atlas: Focado em promover conhecimento e engajamento na resolução de enigmas e problemas lógicos. Foi fundado por você e pelo One Person em 2021, mas o status atual é inativo.
#traveller: Criado por você, este projeto traz um sistema que incentiva a resolução de problemas e propõe uma nova forma de interagir com criptografias. É o sucessor do Cryptographer.
#who-riddle: Um quebra-cabeça bem engenhoso, também fundado por você em 2020, que usa truques inusuais para testar a capacidade de pensar fora da caixa. O status dele é desconhecido.

Espero que isso ajude a se situar! Se precisar de mais detalhes sobre algum deles, é só dar um toque.
```

---

# O Problema do Link de 9 de Fevereiro
O teste consiste em pedir para a Sophia confirmar se o Openrosen mencionou de quem era uma musica ao enviar um link de YouTube no canal de comandos, e depois fazer um follow-up imediato no mesmo contexto.

### Objetivos:
- Testar recuperacao temporal com data natural em portugues ("9 de fevereiro").
- Testar composicao de ferramentas para escopo por canal + autor + periodo.
- Testar resposta baseada em evidencia forte de mensagem (sem inventar).
- Testar follow-up sem reconsulta desnecessaria quando o contexto ja foi resolvido.
- Testar continuidade conversacional logo apos uma resposta de pesquisa.

### Prompt principal:
```
@Sophia 9 de fevereiro o openrosen mandou um link do youtube de uma musica para o canal de comandos. ele chegou a referir de quem era?
```

### Resposta esperada (aproximadamente):
```
Sim, ele comentou sim. Logo depois do link, ele falou da qualidade do M4rkim e mencionou isso diretamente.
```

### Follow-up de continuidade- depende muito da resposta anterior, mas o ideal seria algo como:
```
kkkkkkkk vdd eu lembro. ele diz que foi assistir bem "ecletico" sei la como escreve.
```

### Resposta esperada no follow-up (aproximadamente)
```
Pois e, "cetico". E ele realmente estava impressionado com a qualidade.
```

---

# O Problema dos Dois Glonos
O teste consiste em pedir para a Sophia identificar o "Glonos verdadeiro" quando existem dois membros com o mesmo vulgo, em um contexto de alta criticidade.

### Objetivos:
- Testar deteccao de ambiguidade de identidade no servidor.
- Testar escalacao para desambiguacao (perfil de membro) em vez de parar cedo.
- Testar qualidade de fallback quando a confianca esta insuficiente.
- Garantir que a resposta seja especifica (nao generica) e orientada a decisao.

### Prompt:
```
Olá, @Sophia. Eu estou meio confuso tem dois seres com o vulgo Glonos e agora que lascou de vez. Como eu posso saber qual é o verdadeiro? Eu preciso mandar uma mensagem muito importante para o verdadeiro, é uma questão de vida ou morte, então, por favor, não erre! 
```

### Resposta esperada (comportamento):
```
O verdadeiro e o @subjectless, porque ele e o perfil mais antigo e com historico consistente no servidor.
Se for urgente, manda para esse primeiro.
```

---

# Mudança de objetivo + capacidade de obter mensagens de n sujetos diferentes (no caso 1 aqui)

@Sophia#9017 o que tem no #silksong?

...

@Sophia pega as 10 mensagens do markov mais recentes na chat de #discussão e faz um ranking das mais engraçadas